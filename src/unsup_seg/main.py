import os
import random
import socket
from argparse import Namespace
from distutils.dir_util import copy_tree

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.backends import cudnn

from solver import Solver

torch.autograd.set_detect_anomaly(True)


@hydra.main(config_path="conf/config.yaml", strict=False)
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"running in: {os.getcwd()}")
    cfg.wd = os.getcwd()
    cfg.host = socket.gethostname()
    cfg.project = "default" if not hasattr(cfg, "project") else cfg.project
    cfg = Namespace(**dict(cfg))

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor=cfg.early_stop_metric,
        mode=cfg.early_stop_mode,
        prefix="",
    )

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=None,
        distributed_backend="dp",
        show_progress_bar=True,
        num_sanity_val_steps=0,
        track_grad_norm=2,
        print_nan_grads=True,
        gpus=cfg.gpus,
        gradient_clip_val=cfg.grad_clip,
        val_check_interval=cfg.val_check_interval,
        fast_dev_run=cfg.dev_run,
        max_epochs=cfg.epochs,
    )

    if cfg.ckpt is not None:
        ckpt = cfg.ckpt
    else:
        solver = Solver(cfg)
        trainer.fit(solver)
        ckpt = solver.get_ckpt_path()

    print(f"running test on ckpt: {ckpt}")
    print(f"testing for {cfg.data.upper()}")
    solver = Solver.load_from_checkpoint(ckpt)

    # override checkpoint paths with current conf paths
    solver.hp.timit_path = cfg.timit_path
    solver.hp.buckeye_path = cfg.buckeye_path
    solver.hp.libri_path = cfg.libri_path
    solver.hp.data = cfg.data
    trainer.test(solver)


if __name__ == "__main__":
    main()
