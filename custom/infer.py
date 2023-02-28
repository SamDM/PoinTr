import argparse
import json
from copy import deepcopy

import pydantic
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import open3d as o3d
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


@dataclass
class AdaPoinTrInfer:
    model_path: Path

    def __post_init__(self):
        state_dict = torch.base.load(self.model_path)
        model_cfg = ModelLoader.Cfg(out_n_points=state_dict['config']['out_n_points'])
        self.model = ModelLoader(cfg=model_cfg).get_adapointr_model()
        self.model = cast(AdaPoinTr, torch.nn.DataParallel(self.model).cuda())
        self.model.load_state_dict(state_dict['weights'])
        self.model.eval()

    def infer_tensor(self, xyz: torch.base.Tensor) -> torch.base.Tensor:
        xyz = xyz.float().cuda()
        xyz = torch.base.unsqueeze(xyz, 0)
        with torch.base.no_grad():
            sparse, dense = self.model(xyz)
        out = dense[0]
        return out

    def infer_numpy(self, xyz: np.ndarray) -> np.ndarray:
        xyz = torch.base.tensor(xyz)
        out = self.infer_tensor(xyz)
        out = out.detach().cpu().numpy()
        return out

    def infer_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        xyz = np.asarray(pcd.points)
        out = self.infer_numpy(xyz)
        pcd = tdc.Arr(out).to_o3d_pcd()
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
        }
    )
    config_fpath = out_fpath / "config.json"

    with open(config_fpath, 'w') as fh:
        json.dump(config, fh)

    sys.argv.clear()
    sys.argv.extend(['infer', str(config_fpath)])
    _main()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath", type=Path)
    args = parser.parse_args()
    config_fpath = Path(args.config_fpath)

    class MainCfg(CfgModel):
        weight_fpath: Path
        max_n_points: int
        inputs_outputs: dict[Path, Path]

    cfg = pydantic.parse_file_as(MainCfg, config_fpath)

    model = AdaPoinTrInfer(cfg.weight_fpath)

    for inp_fpath, out_fpath in tqdm(cfg.inputs_outputs.items()):
        inp_pcd = o3d.io.read_point_cloud(str(inp_fpath))
        inp_pcd_dn = deepcopy(inp_pcd)
        downsample_size = 0.05
        while len(inp_pcd_dn.points) > cfg.max_n_points:
            inp_pcd_dn = inp_pcd.voxel_down_sample(downsample_size)
            downsample_size *= 2
        out_pcd = model.infer_pcd(inp_pcd_dn)
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_fpath), out_pcd)


if __name__ == '__main__':
    # _dev()
    _main()
