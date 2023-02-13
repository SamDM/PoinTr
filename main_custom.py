import argparse
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import open3d
# noinspection PyPackageRequirements
import pydantic
import tdkit_core as tdc
from devtools import debug
from tqdm import tqdm

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.AdaPoinTr import AdaPoinTr
from tools import builder
from utils.config import cfg_from_yaml_file


# noinspection PyPackageRequirements,PyPep8Naming
class torch:
    import torch as base
    from torch import nn, optim
    from torch.utils import data
    from torch.utils import tensorboard


REPO_DIR = Path(__file__).parent


class Config(pydantic.BaseModel):

    class DataCfg(pydantic.BaseModel):
        corr_tpath: Path
        true_tpath: Path
        sample_ids: list[str]
        corr_n_points: int = 2048
        true_n_points: int = 16384

    class TrainCfg(pydantic.BaseModel):
        n_epochs: int
        batch_size: int = 1
        n_dataloader_workers: int = 0
        val_every: int = 1

    class ValCfg(pydantic.BaseModel):
        batch_size: int = 1
        n_dataloader_workers: int = 0
        n_logged_point_clouds: int = 1

    pre_trained: bool

    train_data_cfg: DataCfg
    train_cfg: TrainCfg

    val_data_cfg: DataCfg
    val_cfg: ValCfg

    exp_dpath: Path | None = None


def get_adapointr_config():
    adapointr_cfg_file = REPO_DIR / "cfgs/PCN_models/AdaPoinTr.yaml"
    config = cfg_from_yaml_file(adapointr_cfg_file)
    return config


def get_adapointr_model(pre_trained: bool = False) -> AdaPoinTr:
    config = get_adapointr_config()
    model: AdaPoinTr = builder.model_builder(config.model)

    if pre_trained:
        ckpt_path = REPO_DIR.parent / "PoinTr_data/ckpts/AdaPoinTr_PCN.pth"
        state_dict = torch.base.load(ckpt_path)
        model_dict = state_dict["base_model"]
        model.load_state_dict(model_dict, strict=True)

    return model


def get_adapointr_optimizer(model: AdaPoinTr):
    config = get_adapointr_config()
    optimizer = builder.build_optimizer(model, config)
    return optimizer


def get_adapointr_scheduler(model: AdaPoinTr, optimizer: torch.optim.Optimizer):
    config = get_adapointr_config()
    scheduler = builder.build_scheduler(model, optimizer, config, last_epoch=-1)
    return scheduler


@dataclass
class TrainLoop:
    model: AdaPoinTr
    train_cfg: Config.TrainCfg
    train_data_set: torch.data.Dataset
    val_data_set: torch.data.Dataset | None = None
    test_data_set: torch.data.Dataset | None = None
    val_cfg: Config.ValCfg | None = None
    exp_dpath: Path | None = None

    def __post_init__(self):
        assert isinstance(self.train_cfg, Config.TrainCfg)
        assert isinstance(self.train_data_set, torch.data.Dataset)

        if self.val_cfg is not None:
            assert isinstance(self.val_cfg, Config.ValCfg)
            assert isinstance(self.val_data_set, torch.data.Dataset)

        self.model = cast(AdaPoinTr, torch.nn.DataParallel(self.model).cuda())
        self.optimizer = get_adapointr_optimizer(self.model)
        self.schedulers = get_adapointr_scheduler(self.model, self.optimizer)

        self.writer = None
        if self.exp_dpath:
            self.writer = torch.tensorboard.SummaryWriter(str(self.exp_dpath))
        self.epoch = None

    def run_training(self):

        for self.epoch in range(self.train_cfg.n_epochs):
            self._train_one_epoch()

            if self.val_data_set is not None and (self.epoch % self.train_cfg.val_every) == 0:
                validate(self.model, self.val_data_set, self.val_cfg, self.writer, global_step=self.epoch)

    def _train_one_epoch(self):
        train_data_loader = torch.data.DataLoader(
            self.train_data_set, batch_size=self.train_cfg.batch_size, shuffle=True,
            num_workers=self.train_cfg.n_dataloader_workers,
        )

        self.model.train()
        all_metrics = []
        pbar = tqdm(train_data_loader, desc="training")
        for data_idx, (corr, true) in enumerate(pbar):
            corr = corr.cuda()
            true = true.cuda()
            pred = self.model(corr)

            sparse_loss, dense_loss = self.model.module.get_loss(pred, true, self.epoch)
            _loss = sparse_loss + dense_loss
            _loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
            self.optimizer.step()
            self.model.zero_grad()

            # logging
            batch_size = true.shape[0]
            epoch_perc = (self.epoch + 1) / self.train_cfg.n_epochs * 100
            metrics = {
                "sparse/l1": (sparse_loss / batch_size).item(),
                "dense/l1": (dense_loss / batch_size).item(),
            }
            all_metrics.append(metrics)
            pbar.set_description(f"training epoch: {self.epoch:03d} ({epoch_perc:.2f}%)")
            pbar.set_postfix(metrics)

        # epoch done

        for scheduler in self.schedulers:
            scheduler.step()

        # more logging
        metrics = summarize_metrics(all_metrics)
        pbar.set_postfix(metrics)
        pbar.close()

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(tag=f"train/{k}", scalar_value=v, global_step=self.epoch)


def validate(
        model: AdaPoinTr,
        data_set: torch.data.Dataset,
        cfg: Config.ValCfg,
        writer: Path | torch.tensorboard.SummaryWriter | None = None,
        global_step: int = 0,
        **kwargs
):
    chamfer_l1 = ChamferDistanceL1()
    chamfer_l2 = ChamferDistanceL2()

    data_loader = torch.data.DataLoader(
        data_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.n_dataloader_workers,
    )

    model.eval()

    # logging related
    if isinstance(writer, Path):
        writer = torch.tensorboard.SummaryWriter(str(writer))

    pbar = tqdm(data_loader, desc="validating")
    all_metrics = []
    log_pcd_idxs = np.linspace(0, len(data_loader) - 1, num=cfg.n_logged_point_clouds)
    log_pcd_idxs = list(np.unique(np.round(log_pcd_idxs)).astype(int))

    for data_idx, (corr, true) in enumerate(pbar):
        with torch.base.no_grad():
            corr = corr.cuda()
            true = true.cuda()
            pred = model(corr)

        pred_sparse, pred_dense = pred

        def proc_loss(loss):
            batch_size = true.shape[0]
            loss = loss / batch_size
            if isinstance(loss, torch.base.Tensor):
                return loss.item()

        metrics = {
            "sparse/l1": proc_loss(chamfer_l1(pred_sparse, true)),
            "sparse/l2": proc_loss(chamfer_l2(pred_sparse, true)),
            "dense/l1": proc_loss(chamfer_l1(pred_dense, true)),
            "dense/l2": proc_loss(chamfer_l2(pred_dense, true)),
        }
        all_metrics.append(metrics)

        if writer is not None and data_idx in log_pcd_idxs:
            # Monkey-patch torch.utils.tensorboard.SummaryWriter
            from open3d.visualization.tensorboard_plugin import summary
            assert summary
            # Utility function to convert Open3D geometry to a dictionary format
            from open3d.visualization.tensorboard_plugin.util import to_dict_batch

            pcds = [
                ("corr", corr, [1.0, 0.0, 0.0]),
                ("pred_sparse", pred_sparse, [0.0, 0.0, 0.5]),
                ("pred_dense", pred_dense, [0.0, 0.0, 0.0]),
                ("true", true, [0.0, 1.0, 0.0])
            ]

            for name, coords, color in pcds:
                pcd = tdc.Arr(coords.cpu()[0]).to_o3d_pcd(tdc.Arr[float]([color]))
                writer.add_3d(name, to_dict_batch([pcd]), step=global_step)

                # because sometimes the Open3D TB plugin fails me...
                man_dpath = Path(writer.log_dir) / "point_clouds"
                man_dpath.mkdir(exist_ok=True)
                _step = 0 if global_step is None else global_step
                open3d.io.write_point_cloud(str(man_dpath / f"{_step:03d}_{name}.ply"), pcd)

            writer.flush()

    # validation set done

    metrics = summarize_metrics(all_metrics)
    pbar.set_postfix(metrics)
    pbar.close()

    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(tag=f"val/{k}", scalar_value=v, global_step=global_step, **kwargs)


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    d_all = defaultdict(list)
    for d in metrics:
        for k, v in d.items():
            d_all[k].append(v)
    d_summary = {}
    for k, v in d_all.items():
        d_summary[k] = float(np.mean(v).item())
    return d_summary


class DataSet(torch.data.Dataset[tuple[torch.base.Tensor, torch.base.Tensor]]):
    def __init__(self, cfg: Config.DataCfg):
        self.cfg: Config.DataCfg = deepcopy(cfg)

    def __len__(self):
        return len(self.cfg.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.base.Tensor, torch.base.Tensor]:
        sample_id = self.cfg.sample_ids[index]

        def get_pcd(tpath: Path, n_points: int):
            fpath = str(tpath).format(sample_id=sample_id)
            pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(
                fpath, remove_nan_points=True, remove_infinite_points=True)
            pcd = pcd.farthest_point_down_sample(n_points)
            xyz = np.array(pcd.points)
            return torch.base.tensor(xyz, dtype=torch.base.float32)

        corr_pcd = get_pcd(self.cfg.corr_tpath, self.cfg.corr_n_points)
        true_pcd = get_pcd(self.cfg.true_tpath, self.cfg.true_n_points)

        return corr_pcd, true_pcd


def print_model_info(base_model: torch.nn.Module):
    print('Trainable_parameters:')
    print('=' * 25)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print(name)
    print('=' * 25)

    print('Untrainable_parameters:')
    print('=' * 25)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print(name)
    print('=' * 25)


def train(cfg: Config):
    kwargs = dict(model=get_adapointr_model(cfg.pre_trained),
                  exp_dpath=cfg.exp_dpath,
                  train_cfg=cfg.train_cfg,
                  train_data_set=DataSet(cfg.train_data_cfg))

    if cfg.val_cfg:
        kwargs.update(val_cfg=cfg.val_cfg,
                      val_data_set=DataSet(cfg.val_data_cfg))

    train_loop = TrainLoop(**kwargs)
    train_loop.run_training()


def dev():
    data_dpath = Path("/workspace/host/root/media/robovision-syno5-work/nucleus/0039_OCL3D_data/PoC2/pipeline_py_dev/")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dev_out_dpath = Path("/workspace/host/home/Safe/Temp/adapointr/") / timestamp
    dev_out_dpath.mkdir()
    sample_ids = [f"{i:03d}" for i in range(5)]
    n_workers = 0
    n_points = 2**13

    cfg = Config(
        pre_trained=True,
        exp_dpath=dev_out_dpath,

        train_data_cfg=Config.DataCfg(
            corr_tpath=data_dpath / "plant-vox-carve/nrot-003_ncam-003/plant_{sample_id}/carved.ply",
            true_tpath=data_dpath / "ground-truth/plant_{sample_id}/GT.ply",
            sample_ids=sample_ids,
            corr_n_points=n_points,
        ),
        train_cfg=Config.TrainCfg(
            n_epochs=10,
            batch_size=1,
            n_dataloader_workers=n_workers
        ),

        val_data_cfg=Config.DataCfg(
            corr_tpath=data_dpath / "plant-vox-carve/nrot-003_ncam-003/plant_{sample_id}/carved.ply",
            true_tpath=data_dpath / "ground-truth/plant_{sample_id}/GT.ply",
            sample_ids=sample_ids,
            corr_n_points=n_points,
        ),
        val_cfg=Config.ValCfg(
            batch_size=1,
            n_dataloader_workers=n_workers,
        )
    )
    train(cfg)

    # train_data_set = DataSet(cfg.train_data)
    # import tdkit_core as tdc
    # corr, true = map(tdc.Arr, train_data_set[0])
    # open3d.io.write_point_cloud(str(dev_out_dpath / "corr.ply"), corr.to_o3d_pcd())
    # open3d.io.write_point_cloud(str(dev_out_dpath / "true.ply"), true.to_o3d_pcd())


def main():
    root_dir = Path(__file__).parent
    if root_dir not in [Path(p) for p in sys.path]:
        print("extending sys.path")
        sys.path.append(str(root_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath", type=Path)
    args = parser.parse_args()
    cfg = pydantic.parse_file_as(Config, args.config_fpath)
    debug(cfg)
    train(cfg)


if __name__ == '__main__':
    # dev()
    main()
