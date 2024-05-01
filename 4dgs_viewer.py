#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from gaussian_renderer import render, network_gui
import sys
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from scene import GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def viewing(dataset, pipe, checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d):
    gaussians = GaussianModel(
        dataset.sh_degree, 
        gaussian_dim=gaussian_dim, 
        time_duration=time_duration, 
        rot_4d=rot_4d, 
        force_sh_3d=force_sh_3d, 
        sh_degree_t=2 if pipe.eval_shfs_4d else 0
    )

    if checkpoint:
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore(model_params, None)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Listening")

    while True:
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, _, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # print("Rendered frame at camera, sending")
                network_gui.send(net_image_bytes, dataset.source_path)
                if not keep_alive:
                    break
            except Exception:
                network_gui.conn = None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)

    parser.add_argument("--config", type=str)
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)

    # Start GUI server, configure and run data exchange code
    network_gui.init(args.ip, args.port)
    viewing(
        lp.extract(args), 
        pp.extract(args), 
        args.start_checkpoint, 
        args.gaussian_dim, 
        args.time_duration, 
        args.rot_4d, 
        args.force_sh_3d
    )
