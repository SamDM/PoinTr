import argparse
import pandas as pd
import json
import sys
import time
from collections import defaultdict
from contextlib import ContextDecorator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import open3d as o3d
import pydantic
import tdkit_core as tdc
from tqdm import tqdm

_REPO_DIR = Path(__file__).parent.parent
if not _REPO_DIR.name == "PoinTr":
    raise AssertionError(_REPO_DIR)
if _REPO_DIR not in [Path(p) for p in sys.path]:
    print(f"extending sys.path with: {_REPO_DIR}")
    sys.path.append(str(_REPO_DIR))


from custom.train import ModelLoader, torch, CfgModel
from models.AdaPoinTr import AdaPoinTr


class timed(ContextDecorator):
    def __init__(self):
        self.start: float | None = None
        self.end: float | None = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.end = time.time()

    @property
    def delta(self) -> float:
        return self.end - self.start


@dataclass
class AdaPoinTrInfer:
    model_path: Path
    inference_stats_history: dict = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self):
        state_dict = torch.base.load(self.model_path)
        model_cfg = ModelLoader.Cfg(out_n_points=state_dict['config']['out_n_points'])
        self.model = ModelLoader(cfg=model_cfg).get_adapointr_model()
        self.model = cast(AdaPoinTr, torch.nn.DataParallel(self.model).cuda())
        self.model.load_state_dict(state_dict['weights'])
        self.model.eval()

    @staticmethod
    def _sample_name(sample_name: str | None):
        if sample_name is None:
            return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        else:
            return sample_name

    def get_last_history_entry(self) -> tuple[str, dict[str, float]]:
        return next(reversed(self.inference_stats_history.items()))

    def get_history_as_df(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(list(self.inference_stats_history.values()))
        df.index = list(self.inference_stats_history.keys())
        df.index.name = "sample"
        return df

    def infer_tensor(self, xyz: torch.base.Tensor, sample_name: str | None = None) -> torch.base.Tensor:
        sample_name = AdaPoinTrInfer._sample_name(sample_name)

        xyz = torch.base.unsqueeze(xyz, 0)
        with torch.base.no_grad(), timed() as t_forward:
            sparse, dense = self.model(xyz)
        out = dense[0]

        self.inference_stats_history[sample_name]['forward_pass'] = t_forward.delta
        self.inference_stats_history[sample_name]['n_points'] = xyz[0].shape[0]
        return out

    def infer_numpy(self, xyz: np.ndarray, sample_name: str | None = None) -> np.ndarray:
        sample_name = AdaPoinTrInfer._sample_name(sample_name)

        with timed() as t_preproc:
            xyz = torch.base.tensor(xyz)
            xyz = xyz.float().cuda()
            torch.base.cuda.synchronize()
        out = self.infer_tensor(xyz, sample_name=sample_name)
        with timed() as t_postproc:
            out = out.detach().cpu().numpy()

        self.inference_stats_history[sample_name]['np_to_cuda'] = t_preproc.delta
        self.inference_stats_history[sample_name]['cuda_to_np'] = t_postproc.delta
        return out

    def infer_pcd(self, pcd: o3d.geometry.PointCloud, sample_name: str | None = None) -> o3d.geometry.PointCloud:
        sample_name = AdaPoinTrInfer._sample_name(sample_name)

        with timed() as t_preproc:
            xyz = np.asarray(pcd.points)
        out = self.infer_numpy(xyz, sample_name=sample_name)
        with timed() as t_postproc:
            pcd = tdc.Arr(out).to_o3d_pcd()

        self.inference_stats_history[sample_name]['pcd_to_np'] = t_preproc.delta
        self.inference_stats_history[sample_name]['np_to_pcd'] = t_postproc.delta
        return pcd


def _dev():
    data_dpath = Path("/workspace/host/root/media/robovision-syno5-work/nucleus/0039_OCL3D_data/PoC2/pipeline_py_dev/")
    weights_rel_fpath = Path("pcdcorr_adapointr/plant-vox-carve/nrot-003_ncam-003/"
                             "train.back.3/training_history/checkpoints/last/model.pth")
    carved_rel_fpath = Path("plant-vox-carve/nrot-003_ncam-003/")

    out_fpath = Path("/workspace/host/home/Safe/Temp/AdaPoinTr_infer")

    config = dict(
        weight_fpath=str(data_dpath / weights_rel_fpath),
        max_n_points=2**14,
        inputs_outputs={
            str(data_dpath / carved_rel_fpath / f"plant_{i:03d}" / "carved.ply"):
            str(out_fpath / f"out_{i:03d}.ply")
            for i in range(5)
        },
        stats_out_fpath=str(out_fpath / "stats.tsv")
    )
    config_fpath = out_fpath / "config.json"

    with open(config_fpath, 'w') as fh:
        json.dump(config, fh)

    sys.argv.clear()
    sys.argv.extend(['infer', str(config_fpath)])
    _main()


def _main():
    class _MainCfg(CfgModel):
        weight_fpath: Path
        max_n_points: int
        inputs_outputs: dict[Path, Path]
        stats_out_fpath: Path

    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath", type=Path)
    args = parser.parse_args()
    config_fpath = Path(args.config_fpath)

    cfg = pydantic.parse_file_as(_MainCfg, config_fpath)

    model = AdaPoinTrInfer(cfg.weight_fpath)

    pbar = tqdm(cfg.inputs_outputs.items())
    for inp_fpath, out_fpath in pbar:
        inp_pcd = o3d.io.read_point_cloud(str(inp_fpath))
        inp_pcd_dn = deepcopy(inp_pcd)
        downsample_size = 0.05
        while len(inp_pcd_dn.points) > cfg.max_n_points:
            inp_pcd_dn = inp_pcd.voxel_down_sample(downsample_size)
            downsample_size *= 1.5
        out_pcd = model.infer_pcd(inp_pcd_dn)
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_fpath), out_pcd)

        sample_name, stats = model.get_last_history_entry()
        stats["downsample_size"] = downsample_size
        pbar.set_postfix(stats)

    df = model.get_history_as_df()
    df.to_csv(cfg.stats_out_fpath, sep="\t")


if __name__ == '__main__':
    # _dev()
    _main()
