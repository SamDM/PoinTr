import argparse
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import open3d as o3d
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
    from torch.utils import tensorboard as tb

    # Monkey-patch torch.utils.tensorboard.SummaryWriter
    from open3d.visualization.tensorboard_plugin import summary as _summary
    assert _summary

    # Utility functions to send Open3D geometry to tensorboard
    from open3d.visualization.tensorboard_plugin import util as tb_o3d


@dataclass
class ModelLoader:
    class Cfg(pydantic.BaseModel):
        pre_trained: bool
        out_n_points: int = 2**14

    cfg: Cfg

    _REPO_DIR: Path = Path(__file__).parent.parent
    assert _REPO_DIR.name == "PoinTr"

    def get_adapointr_config(self):
        adapointr_cfg_file = self._REPO_DIR / "cfgs/PCN_models/AdaPoinTr.yaml"
        default_config = cfg_from_yaml_file(adapointr_cfg_file)
        default_config.model.num_points = self.cfg.out_n_points
        return default_config

    def get_adapointr_model(self) -> AdaPoinTr:
        adapointr_config = self.get_adapointr_config()
        model: AdaPoinTr = builder.model_builder(adapointr_config.model)

        if self.cfg.pre_trained:
            ckpt_path = self._REPO_DIR.parent / "PoinTr_data/ckpts/AdaPoinTr_PCN.pth"
            state_dict = torch.base.load(ckpt_path)
            model_dict = state_dict["base_model"]
            # todo: replace non-matching keys
            model.load_state_dict(model_dict, strict=True)

        return model

    def get_adapointr_optimizer(self, model: AdaPoinTr):
        config = self.get_adapointr_config()
        optimizer = builder.build_optimizer(model, config)
        return optimizer

    def get_adapointr_scheduler(self, model: AdaPoinTr, optimizer: torch.optim.Optimizer):
        config = self.get_adapointr_config()
        scheduler = builder.build_scheduler(model, optimizer, config, last_epoch=-1)
        return scheduler


@dataclass
class TrainLoop:
    class Cfg(pydantic.BaseModel):
        train_n_epochs: int
        train_batch_size: int = 1
        train_n_dataloader_workers: int = 0

        val_at_least_every_n_epochs: int
        val_keep_at_most_n_checkpoints: int = 100
        val_log_at_most_n_point_clouds: int = 10
        val_batch_size: int = 1

        exp_dpath: Path

    train_cfg: Cfg
    model_cfg: ModelLoader.Cfg
    train_data_set: torch.data.Dataset
    val_data_set: torch.data.Dataset | None = None

    def __post_init__(self):
        model_loader = ModelLoader(self.model_cfg)
        model = model_loader.get_adapointr_model()
        self.model = cast(AdaPoinTr, torch.nn.DataParallel(model).cuda())
        self.optimizer = model_loader.get_adapointr_optimizer(self.model)
        self.schedulers = model_loader.get_adapointr_scheduler(self.model, self.optimizer)

        self.writer = torch.tb.SummaryWriter(str(self.train_cfg.exp_dpath / "training_history/tensorboard"))
        self.epoch = None

    def run_training(self):

        for self.epoch in range(self.train_cfg.train_n_epochs):
            self._train_one_epoch()

            if self.val_data_set is not None and (self.epoch % self.train_cfg.val_at_least_every_n_epochs) == 0:
                Validate(
                    cfg=Validate.Cfg(
                        global_step=self.epoch,
                        batch_size=self.train_cfg.val_batch_size,
                        n_dataloader_workers=self.train_cfg.val_batch_size,
                        log_at_most_n_point_clouds=self.train_cfg.val_log_at_most_n_point_clouds
                    ),
                    model=self.model,
                    writer=self.writer,
                    data_set=self.val_data_set
                ).run()

    def _train_one_epoch(self):
        train_data_loader = torch.data.DataLoader(
            self.train_data_set, batch_size=self.train_cfg.train_batch_size, shuffle=True,
            num_workers=self.train_cfg.train_n_dataloader_workers
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
            epoch_perc = (self.epoch + 1) / self.train_cfg.train_n_epochs * 100
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
        pbar.refresh()
        pbar.close()

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(tag=f"train/{k}", scalar_value=v, global_step=self.epoch)


@dataclass
class Validate:

    class Cfg(pydantic.BaseModel):
        global_step: int = 0
        batch_size: int = 1
        n_dataloader_workers: int = 0
        log_at_most_n_point_clouds: int = 10

    cfg: Cfg
    model: AdaPoinTr
    data_set: torch.data.Dataset
    writer: torch.tb.SummaryWriter | None = None,

    def run(self):
        chamfer_l1 = ChamferDistanceL1()
        chamfer_l2 = ChamferDistanceL2()

        data_loader = torch.data.DataLoader(
            self.data_set, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.n_dataloader_workers,
        )

        self.model.eval()

        pbar = tqdm(data_loader, desc="validating")
        all_metrics = []
        log_pcd_idxs = np.linspace(0, len(data_loader) - 1, num=self.cfg.log_at_most_n_point_clouds)
        log_pcd_idxs = list(np.unique(np.round(log_pcd_idxs)).astype(int))

        for data_idx, (corr, true) in enumerate(pbar):
            with torch.base.no_grad():
                corr = corr.cuda()
                true = true.cuda()
                pred = self.model(corr)

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
            pbar.set_postfix(metrics)

            if data_idx in log_pcd_idxs:

                pcds = [
                    ("corr", corr, [1.0, 0.0, 0.0]),
                    ("pred_sparse", pred_sparse, [0.0, 0.0, 0.5]),
                    ("pred_dense", pred_dense, [0.0, 0.0, 0.0]),
                    ("true", true, [0.0, 1.0, 0.0])
                ]

                for name, coords, color in pcds:
                    pcd = tdc.Arr(coords.cpu()[0]).to_o3d_pcd(tdc.Arr[float]([color]))
                    # noinspection PyUnresolvedReferences
                    self.writer.add_3d(name, torch.tb_o3d.to_dict_batch([pcd]), step=self.cfg.global_step)

                    # because I also like standalone PCD viewers.
                    man_dpath = Path(self.writer.log_dir).parent / "point_clouds"
                    man_dpath.mkdir(exist_ok=True)
                    o3d.io.write_point_cloud(str(man_dpath / f"{self.cfg.global_step:03d}_{name}.ply"), pcd)

                self.writer.flush()

        # validation set done

        metrics = summarize_metrics(all_metrics)
        pbar.set_postfix(metrics)
        pbar.refresh()
        pbar.close()

        for k, v in metrics.items():
            self.writer.add_scalar(tag=f"val/{k}", scalar_value=v, global_step=self.cfg.global_step)


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
    class Cfg(pydantic.BaseModel):
        sample_ids: list[str]
        corr_tpath: Path
        true_tpath: Path
        cache_dir: Path
        inp_n_points: int = 2**13
        out_n_points: int = 2**14

    def __init__(self, cfg: Cfg):
        self.cfg: DataSet.Cfg = deepcopy(cfg)

    def __len__(self):
        return len(self.cfg.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.base.Tensor, torch.base.Tensor]:
        sample_id = self.cfg.sample_ids[index]

        def get_pcd(tpath: Path, desired_n_points: int, cache_dir: Path):
            full_fpath = str(tpath).format(sample_id=sample_id)
            cache_fpath = cache_dir / f"{index:04d}.ply"

            pcd = self._load_cached_pcd(Path(full_fpath), cache_fpath, desired_n_points)
            xyz = np.array(pcd.points)
            return torch.base.tensor(xyz, dtype=torch.base.float32)

        corr_pcd = get_pcd(self.cfg.corr_tpath, self.cfg.inp_n_points, self.cfg.cache_dir / "corr")
        true_pcd = get_pcd(self.cfg.true_tpath, self.cfg.out_n_points, self.cfg.cache_dir / "true")

        return corr_pcd, true_pcd

    @staticmethod
    def _find_voxel_size(pcd: o3d.geometry.PointCloud, desired_n_points: int):

        idxs = np.random.choice(len(pcd.points), 100, replace=False)
        tiny_pcd = pcd.select_by_index(idxs)
        volume = np.prod(tiny_pcd.get_max_bound() - tiny_pcd.get_min_bound())
        vox_size = volume/desired_n_points
        while True:
            dn_pcd = pcd.voxel_down_sample(vox_size)
            # print(f"{vox_size:5g}: {len(pcd.points):10d} -> "
            #       f"{len(dn_pcd.points):10d} (desired: {desired_n_points:10d})")
            if len(dn_pcd.points) > desired_n_points * 2:
                vox_size *= np.power(2, 1/3)
            elif len(dn_pcd.points) < desired_n_points:
                vox_size *= np.power(0.9, 1/3)
            else:
                # print("OK!")
                return vox_size

    def _load_cached_pcd(self, full_fpath: Path, cache_fpath: Path, desired_n_points: int) -> o3d.geometry.PointCloud:
        if not cache_fpath.is_file():
            full_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
                str(full_fpath), remove_nan_points=True, remove_infinite_points=True)

            # Voxel down-sample to have uniform data representation
            vox_size = self._find_voxel_size(full_pcd, desired_n_points)
            dn_pcd = full_pcd.voxel_down_sample(vox_size)

            cache_fpath.parent.mkdir(exist_ok=True, parents=True)
            o3d.io.write_point_cloud(str(cache_fpath), dn_pcd)

        else:
            dn_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
                str(cache_fpath), remove_nan_points=True, remove_infinite_points=True)

        # Result will have slightly too many points,
        # in those points, select randomly the desired amount
        idxs = np.random.choice(len(dn_pcd.points), desired_n_points, replace=False)
        return dn_pcd.select_by_index(idxs)


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


class Train:
    class Cfg(pydantic.BaseModel):
        train_n_epochs: int
        train_batch_size: int = 1
        train_n_dataloader_workers: int = 0
        train_sample_ids: list[str]

        val_at_least_every_n_epochs: int
        val_keep_at_most_n_checkpoints: int = 100
        val_log_at_most_n_point_clouds: int = 10
        val_batch_size: int = 1
        val_n_dataloader_workers: int = 0
        val_sample_ids: list[str]

        inp_n_points: int = 2**13
        model_cfg: ModelLoader.Cfg

        exp_dpath: Path
        corr_tpath: Path
        true_tpath: Path

    def __init__(self, cfg: Cfg):
        self.cfg: Train.Cfg = cfg

        self.train_loop = TrainLoop(
            train_cfg=TrainLoop.Cfg(
                train_n_epochs=self.cfg.train_n_epochs,
                train_batch_size=self.cfg.train_batch_size,
                train_n_dataloader_workers=self.cfg.train_n_dataloader_workers,

                val_at_least_every_n_epochs=self.cfg.val_at_least_every_n_epochs,
                val_keep_at_most_n_checkpoints=self.cfg.val_keep_at_most_n_checkpoints,
                val_log_at_most_n_point_clouds=self.cfg.val_log_at_most_n_point_clouds,
                val_batch_size=self.cfg.val_batch_size,

                exp_dpath=self.cfg.exp_dpath,
            ),
            model_cfg=self.cfg.model_cfg,
            train_data_set=DataSet(cfg=DataSet.Cfg(
                sample_ids=self.cfg.train_sample_ids,
                corr_tpath=self.cfg.corr_tpath,
                true_tpath=self.cfg.true_tpath,
                cache_dir=self.cfg.exp_dpath / "cache/train",
                n_points=self.cfg.inp_n_points
            )),
            val_data_set=DataSet(cfg=DataSet.Cfg(
                sample_ids=self.cfg.val_sample_ids,
                corr_tpath=self.cfg.corr_tpath,
                true_tpath=self.cfg.true_tpath,
                cache_dir=self.cfg.exp_dpath / "cache/val",
                n_points=self.cfg.inp_n_points
            )),
        )

    def run(self):
        self.train_loop.run_training()


def _dev():
    data_dpath = Path("/workspace/host/root/media/robovision-syno5-work/nucleus/0039_OCL3D_data/PoC2/pipeline_py_dev/")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dev_out_dpath = Path("/workspace/host/home/Safe/Proj/Number/"
                         "0039_Phenotyping-with-occlusion/PoC2/"
                         "src/0039_3d-recon-benchmark/PoC2/_dev_outputs/adapointr") / timestamp
    dev_out_dpath.mkdir(parents=True)
    sample_ids = [f"{i:03d}" for i in range(5)]
    batch_size = 1
    n_workers = 0
    inp_n_points = 2**13
    out_n_points = 2**14

    train = Train(cfg=Train.Cfg(
        train_n_epochs=100,
        train_batch_size=batch_size,
        train_n_dataloader_workers=n_workers,
        train_sample_ids=sample_ids[:3],

        val_at_least_every_n_epochs=10,
        val_keep_at_most_n_checkpoints=10,
        val_log_at_most_n_point_clouds=10,
        val_batch_size=batch_size,
        val_n_dataloader_workers=n_workers,
        val_sample_ids=sample_ids[3:],

        inp_n_points=inp_n_points,
        model_cfg=ModelLoader.Cfg(
            pre_trained=True,
            n_points=out_n_points,
        ),

        exp_dpath=dev_out_dpath,
        corr_tpath=data_dpath / "plant-vox-carve/nrot-003_ncam-003/plant_{sample_id}/carved.ply",
        true_tpath=data_dpath / "ground-truth/plant_{sample_id}/colored.ply",
    ))
    train.run()


def _main():
    root_dir = Path(__file__).parent
    if root_dir not in [Path(p) for p in sys.path]:
        print("extending sys.path")
        sys.path.append(str(root_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath", type=Path)
    args = parser.parse_args()
    cfg = pydantic.parse_file_as(Train.Cfg, args.config_fpath)
    debug(cfg)
    Train(cfg=cfg).run()


if __name__ == '__main__':
    _dev()
    # main()
