import argparse
import shutil
import json
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

_REPO_DIR = Path(__file__).parent.parent
assert _REPO_DIR.name == "PoinTr"
if _REPO_DIR not in [Path(p) for p in sys.path]:
    print(f"extending sys.path with: {_REPO_DIR}")
    sys.path.append(str(_REPO_DIR))

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


class CfgModel(pydantic.BaseModel):
    class Config:
        extra = "forbid"


@dataclass
class ModelLoader:
    class Cfg(CfgModel):
        pre_trained: bool
        out_n_points: int = 2**14

    cfg: Cfg

    def get_adapointr_config(self):
        adapointr_cfg_file = _REPO_DIR / "cfgs/PCN_models/AdaPoinTr.yaml"
        default_config = cfg_from_yaml_file(adapointr_cfg_file)
        default_config.model.num_points = self.cfg.out_n_points
        return default_config

    def get_adapointr_model(self) -> AdaPoinTr:
        adapointr_config = self.get_adapointr_config()
        model: AdaPoinTr = builder.model_builder(adapointr_config.model)

        if self.cfg.pre_trained:
            ckpt_path = _REPO_DIR.parent / "PoinTr_data/ckpts/AdaPoinTr_PCN.pth"
            state_dict = torch.base.load(ckpt_path)
            model_dict = state_dict["base_model"]
            # todo: replace non-matching keys
            model.load_state_dict(model_dict, strict=True)

        return model

    def get_adapointr_optimizer(self, model: AdaPoinTr):
        config = self.get_adapointr_config()
        optimizer = builder.build_optimizer(model, config)
        return optimizer

    def get_adapointr_schedulers(self, model: AdaPoinTr, optimizer: torch.optim.Optimizer):
        config = self.get_adapointr_config()
        scheduler = builder.build_scheduler(model, optimizer, config, last_epoch=-1)
        return scheduler


@dataclass
class TrainLoop:
    class Cfg(CfgModel):
        train_n_epochs: int
        train_log_at_most_n_point_clouds: int = 10
        train_batch_size: int = 1
        train_n_dataloader_workers: int = 0
        train_shuffle_data: bool = True

        val_every_n_epochs: int
        val_keep_at_most_n_epochs: int = 100
        val_log_at_most_n_point_clouds: int = 10
        val_batch_size: int = 1
        val_n_dataloader_workers: int = 0

        exp_dpath: Path

    train_cfg: Cfg
    model_cfg: ModelLoader.Cfg
    train_data_set: torch.data.Dataset
    val_data_set: torch.data.Dataset

    def __post_init__(self):
        model_loader = ModelLoader(self.model_cfg)
        model = model_loader.get_adapointr_model()
        self.model = cast(AdaPoinTr, torch.nn.DataParallel(model).cuda())
        self.optimizer = model_loader.get_adapointr_optimizer(self.model)
        self.schedulers = model_loader.get_adapointr_schedulers(self.model, self.optimizer)

        self.writer = torch.tb.SummaryWriter(str(self.train_cfg.exp_dpath / "training_history/tensorboard"))
        self.epoch: int = 0
        self.last_train_metrics: dict[str, float] | None = None
        self.last_val_metrics: dict[str, float] | None = None
        self.best_val_metrics: dict[str, float] | None = None
        self.best_epoch: int = 0

        self._try_resume()

    def _try_resume(self):
        ckpt_dir = self.train_cfg.exp_dpath / "training_history" / "checkpoints"
        last_ckpt_dir = ckpt_dir / "last"
        best_ckpt_dir = ckpt_dir / "best"

        if (last_ckpt_dir / "state.pth").is_file():
            # load model state
            model_state = torch.base.load(last_ckpt_dir / "model.pth")
            self.model.load_state_dict(model_state['weights'], strict=True)

            # and optimizer + scheduler state
            training_state = torch.base.load(last_ckpt_dir / "state.pth")
            self.optimizer.load_state_dict(training_state['optimizer'])
            for i, scheduler_state in enumerate(training_state['schedulers']):
                self.schedulers[i].load_state_dict(scheduler_state)

            # and metrics
            with open(best_ckpt_dir / "metrics.json") as fh:
                best_metrics = json.load(fh)
                self.best_epoch = best_metrics['epoch']
                self.best_val_metrics = best_metrics['val']
            with open(last_ckpt_dir / "metrics.json") as fh:
                last_metrics = json.load(fh)
                self.epoch = last_metrics['epoch']
                self.last_train_metrics = last_metrics['train']
                self.last_val_metrics = last_metrics['val']

    def run_training(self):

        for self.epoch in range(self.epoch, self.train_cfg.train_n_epochs):
            # noinspection PyAttributeOutsideInit
            self.last_train_metrics = self._train_one_epoch()

            if self._is_val_epoch():
                self._validate()
                self._save_metrics(Path(f"epoch/{self.epoch:03d}"))
                self._update_checkpoints()
                self._cleanup()

    def _is_val_epoch(self) -> bool:
        is_val_epoch = (self.epoch % self.train_cfg.val_every_n_epochs) == 0
        is_final_epoch = self.epoch + 1 == self.train_cfg.train_n_epochs
        return is_val_epoch or is_final_epoch

    def _train_one_epoch(self):
        train_data_loader = torch.data.DataLoader(
            self.train_data_set,
            batch_size=self.train_cfg.train_batch_size,
            shuffle=self.train_cfg.train_shuffle_data,
            num_workers=self.train_cfg.train_n_dataloader_workers
        )
        log_pcd_idxs = get_spaced_idxs(self.train_cfg.train_log_at_most_n_point_clouds, len(train_data_loader))

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

            if data_idx in log_pcd_idxs and self._is_val_epoch():
                log_pcd_data(corr=corr, pred_dense=pred[-1], pred_sparse=pred[0], true=true,
                             sample_name=f"train/{data_idx:03d}",
                             writer=self.writer,
                             global_step=self.epoch)

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

        return metrics

    def _validate(self):
        self.last_val_metrics = Validate(
            cfg=Validate.Cfg(
                global_step=self.epoch,
                batch_size=self.train_cfg.val_batch_size,
                n_dataloader_workers=self.train_cfg.val_n_dataloader_workers,
                log_at_most_n_point_clouds=self.train_cfg.val_log_at_most_n_point_clouds
            ),
            model=self.model,
            writer=self.writer,
            data_set=self.val_data_set
        ).run()

    def _save_metrics(self, where: Path):
        out_fpath = self.train_cfg.exp_dpath / f"training_history" / where / "metrics.json"
        with open(out_fpath, "w") as fh:
            json.dump(dict(
                epoch=self.epoch,
                train=self.last_train_metrics,
                val=self.last_val_metrics,
            ), fh, indent=2)

    def _update_checkpoints(self):
        checkpoints_dir = self.train_cfg.exp_dpath / "training_history/checkpoints"
        checkpoints_dir_last = checkpoints_dir / "last"
        checkpoints_dir_last.mkdir(exist_ok=True, parents=True)

        torch.base.save(dict(
                weights=self.model.state_dict(),
                config=self.model_cfg.dict(),
            ), checkpoints_dir_last / "model.pth")
        torch.base.save(dict(
                optimizer=self.optimizer.state_dict(),
                schedulers=[s.state_dict() for i, s in enumerate(self.schedulers)]
            ), checkpoints_dir_last / "state.pth")
        self._save_metrics(checkpoints_dir_last)

        if self.best_val_metrics is None or self.last_val_metrics['dense/l1'] < self.best_val_metrics['dense/l1']:
            checkpoints_dir_best = checkpoints_dir / "best"
            checkpoints_dir_best.mkdir(exist_ok=True, parents=True)
            checkpoints_dir_best_back = checkpoints_dir / "best.back"

            self.best_epoch = self.epoch
            self.best_val_metrics = self.last_val_metrics

            shutil.move(checkpoints_dir_best, checkpoints_dir_best_back)
            try:
                shutil.copytree(checkpoints_dir_last, checkpoints_dir_best)
            except Exception as e:
                shutil.move(checkpoints_dir_best_back, checkpoints_dir_best)
                raise e
            else:
                shutil.rmtree(checkpoints_dir_best_back)

    def _cleanup(self):
        history = {}
        for epoch_log_dpath in (self.train_cfg.exp_dpath / "training_history" / "epoch").iterdir():
            epoch = int(epoch_log_dpath.name)
            with open(epoch_log_dpath / "metrics.json") as fh:
                loss = json.load(fh)['val']['dense/l1']
            history[epoch] = dict(epoch=epoch, loss=loss, path=epoch_log_dpath)
        history = {k: history[k] for k in sorted(history.keys())}

        n_must_del = len(history) - self.train_cfg.val_keep_at_most_n_epochs
        if n_must_del > 0:
            # remove logs where difference in loss was lowest
            loss_history = [v['loss'] for v in history.values()]
            loss_diff = np.array([
                np.concatenate([[np.nan], np.diff(loss_history)]),
                np.concatenate([np.diff(loss_history), [np.nan]]),
            ])
            loss_diff = np.nanmean(np.abs(loss_diff), axis=0)
            # give prio to keep the first epoch, the best epoch and the last epoch
            has_prio = [epoch in (0, self.best_epoch, self.epoch) for epoch in history.keys()]
            keep_prio = sorted(list(zip(has_prio, loss_diff, history.keys())), reverse=True)

            for (_, _, del_epoch) in keep_prio[self.train_cfg.val_keep_at_most_n_epochs:]:
                del_path = history[del_epoch]["path"]
                shutil.rmtree(del_path)


@dataclass
class Validate:

    class Cfg(CfgModel):
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
        log_pcd_idxs = get_spaced_idxs(self.cfg.log_at_most_n_point_clouds, len(data_loader))

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
                log_pcd_data(
                    corr=corr, pred_sparse=pred_sparse, pred_dense=pred_dense, true=true,
                    sample_name=f"val/{data_idx:03d}",
                    writer=self.writer,
                    global_step=self.cfg.global_step,
                )

        # validation set done

        metrics = summarize_metrics(all_metrics)
        pbar.set_postfix(metrics)
        pbar.refresh()
        pbar.close()

        for k, v in metrics.items():
            self.writer.add_scalar(tag=f"val/{k}", scalar_value=v, global_step=self.cfg.global_step)

        return metrics


def log_pcd_data(
        corr, pred_sparse, pred_dense, true,
        sample_name: str,
        writer: torch.tb.SummaryWriter,
        global_step: int,
):
    pcds = [
        ("corr", corr, [1.0, 0.0, 0.0]),
        ("pred_sparse", pred_sparse, [0.0, 0.0, 0.5]),
        ("pred_dense", pred_dense, [0.0, 0.0, 0.0]),
        ("true", true, [0.0, 1.0, 0.0])
    ]

    for pcd_name, coords, color in pcds:
        if isinstance(coords, torch.base.Tensor):
            coords = coords.detach().cpu().numpy()
        name = f"{sample_name}_{pcd_name}"
        pcd = tdc.Arr(coords[0]).to_o3d_pcd(tdc.Arr[float]([color]))
        # noinspection PyUnresolvedReferences
        writer.add_3d(name, torch.tb_o3d.to_dict_batch([pcd]), step=global_step)

        # because I also like standalone PCD viewers.
        pcd_fpath = (
            Path(writer.log_dir).parent /
            f"epoch/{global_step:03d}/point_clouds" /
            f"{name}.ply"
        )
        pcd_fpath.parent.mkdir(exist_ok=True, parents=True)
        o3d.io.write_point_cloud(str(pcd_fpath), pcd)

    writer.flush()


def get_spaced_idxs(sample_size: int, total_size: int):
    spaced_idxs = np.linspace(0, total_size - 1, num=sample_size)
    spaced_idxs = list(np.unique(np.round(spaced_idxs)).astype(int))
    return spaced_idxs


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    d_all = defaultdict(list)
    for d in metrics:
        for k, v in d.items():
            d_all[k].append(v)
    d_summary = {}
    for k, v in d_all.items():
        d_summary[k] = float(np.mean(v).item())
    return d_summary


@dataclass
class Augment:
    class Cfg(CfgModel):
        rotate_around_z: bool = False
        tilt_deg_std: float = 0.0
        rel_scale_change_std: float = 0.0
        translate_std: float = 0.0

    cfg: Cfg

    def get_augment_fn(self):
        quat_rotate_z = tdc.Quat.rot_z(np.random.uniform(np.random.uniform(0, np.pi * 2)))
        quat_tilt_x = tdc.Quat.rot_x(np.random.normal(0, self.cfg.tilt_deg_std/180*np.pi))
        quat_tilt_y = tdc.Quat.rot_y(np.random.normal(0, self.cfg.tilt_deg_std/180*np.pi))
        scale = np.clip(np.random.normal(1, self.cfg.rel_scale_change_std), 0, 2)
        trans = np.random.normal(0, self.cfg.translate_std, size=3)

        def augment_fn(xyz: np.ndarray):
            if self.cfg.rotate_around_z:
                xyz = quat_rotate_z.rotate_points(xyz, np.array([0.0, 0.0, 0.0]))
            if self.cfg.tilt_deg_std != 0.0:
                xyz = quat_tilt_x.rotate_points(xyz, np.array([0.0, 0.0, 0.0]))
                xyz = quat_tilt_y.rotate_points(xyz, np.array([0.0, 0.0, 0.0]))
            if self.cfg.rel_scale_change_std != 0.0:
                xyz = xyz * scale
            if self.cfg.translate_std != 0.0:
                xyz = xyz + trans
            return xyz

        return augment_fn


class DataSet(torch.data.Dataset[tuple[torch.base.Tensor, torch.base.Tensor]]):
    class Cfg(CfgModel):
        sample_ids: list[str]
        corr_tpath: Path
        true_tpath: Path
        cache_dir: Path
        inp_n_points: int = 2**13
        out_n_points: int = 2**14
        augment_cfg: Augment.Cfg | None = None

    def __init__(self, cfg: Cfg):
        self.cfg: DataSet.Cfg = deepcopy(cfg)

    def __len__(self):
        return len(self.cfg.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.base.Tensor, torch.base.Tensor]:
        sample_id = self.cfg.sample_ids[index]
        if self.cfg.augment_cfg is not None:
            augment_fn = Augment(self.cfg.augment_cfg).get_augment_fn()
        else:
            augment_fn = lambda x: x

        def get_pcd(tpath: Path, desired_n_points: int, cache_dir: Path):
            full_fpath = str(tpath).format(sample_id=sample_id)
            cache_fpath = cache_dir / f"{index:04d}.ply"

            pcd = self._load_cached_pcd(Path(full_fpath), cache_fpath, desired_n_points)
            xyz = np.array(pcd.points)

            xyz = augment_fn(xyz)

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
    class Cfg(CfgModel):
        train_n_epochs: int
        train_log_at_most_n_point_clouds: int = 10
        train_batch_size: int = 1
        train_n_dataloader_workers: int = 0
        train_shuffle_data: bool = True
        train_sample_ids: list[str]
        train_augment_cfg: Augment.Cfg = pydantic.Field(default_factory=Augment.Cfg)

        val_every_n_epochs: int
        val_keep_at_most_n_epochs: int = 100
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
                train_log_at_most_n_point_clouds=self.cfg.train_log_at_most_n_point_clouds,
                train_batch_size=self.cfg.train_batch_size,
                train_n_dataloader_workers=self.cfg.train_n_dataloader_workers,
                train_shuffle_data=self.cfg.train_shuffle_data,

                val_every_n_epochs=self.cfg.val_every_n_epochs,
                val_keep_at_most_n_epochs=self.cfg.val_keep_at_most_n_epochs,
                val_log_at_most_n_point_clouds=self.cfg.val_log_at_most_n_point_clouds,
                val_batch_size=self.cfg.val_batch_size,
                val_n_dataloader_workers=self.cfg.val_n_dataloader_workers,

                exp_dpath=self.cfg.exp_dpath,
            ),
            model_cfg=self.cfg.model_cfg,
            train_data_set=DataSet(cfg=DataSet.Cfg(
                sample_ids=self.cfg.train_sample_ids,
                corr_tpath=self.cfg.corr_tpath,
                true_tpath=self.cfg.true_tpath,
                cache_dir=self.cfg.exp_dpath / "cache/train",
                inp_n_points=self.cfg.inp_n_points,
                augment_cfg=self.cfg.train_augment_cfg,
            )),
            val_data_set=DataSet(cfg=DataSet.Cfg(
                sample_ids=self.cfg.val_sample_ids,
                corr_tpath=self.cfg.corr_tpath,
                true_tpath=self.cfg.true_tpath,
                cache_dir=self.cfg.exp_dpath / "cache/val",
                inp_n_points=self.cfg.inp_n_points,
                augment_cfg=None,
            )),
        )

    def run(self):
        self.train_loop.run_training()


def _dev():
    data_dpath = Path("/workspace/host/root/media/robovision-syno5-work/nucleus/0039_OCL3D_data/PoC2/pipeline_py_dev/")
    run_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_name = "resume-test"
    dev_out_dpath = Path("/workspace/host/home/Safe/Proj/Number/"
                         "0039_Phenotyping-with-occlusion/PoC2/"
                         "src/0039_3d-recon-benchmark/PoC2/_dev_outputs/adapointr") / run_name
    dev_out_dpath.mkdir(parents=True, exist_ok=True)
    sample_ids = [f"{i:03d}" for i in range(5)]
    batch_size = 1
    n_workers = 0
    inp_n_points = 2**13
    out_n_points = 2**14

    train = Train(cfg=Train.Cfg(
        train_n_epochs=200,
        train_log_at_most_n_point_clouds=10,
        train_batch_size=batch_size,
        train_n_dataloader_workers=n_workers,
        train_shuffle_data=False,
        train_sample_ids=sample_ids[:3],
        train_augment_cfg=Augment.Cfg(
            # rotate_around_z=True,
            # tilt_deg_std=10,
            # rel_scale_change_std=0.15,
            # translate_std=0.5,
        ),

        val_every_n_epochs=10,
        val_keep_at_most_n_epochs=4,
        val_log_at_most_n_point_clouds=10,
        val_batch_size=batch_size,
        val_n_dataloader_workers=n_workers,
        val_sample_ids=sample_ids[3:],

        inp_n_points=inp_n_points,
        model_cfg=ModelLoader.Cfg(
            pre_trained=True,
            out_n_points=out_n_points,
        ),

        exp_dpath=dev_out_dpath,
        corr_tpath=data_dpath / "plant-vox-carve/nrot-003_ncam-003/plant_{sample_id}/carved.ply",
        true_tpath=data_dpath / "ground-truth/plant_{sample_id}/colored.ply",
    ))
    train.run()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath", type=Path)
    args = parser.parse_args()
    cfg = pydantic.parse_file_as(Train.Cfg, args.config_fpath)
    debug(cfg)
    Train(cfg=cfg).run()


if __name__ == '__main__':
    _dev()
    # main()
