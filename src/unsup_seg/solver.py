import glob
from collections import OrderedDict, defaultdict

import dill
import wandb

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch_optimizer as optim_extra

from torch import optim
from torch.utils.data import ConcatDataset, DataLoader

from dataloader import (
    LibriSpeechDataset,
    TrainTestDataset,
    TrainValTestDataset,
    collate_fn_padd,
)
from next_frame_classifier import NextFrameClassifier
from utils import (
    PrecisionRecallMetric,
    StatsMeter,
    line,
    max_min_norm,
    replicate_first_k_frames,
)


class Solver(LightningModule):
    def __init__(self, hparams):
        super(Solver, self).__init__()
        hp = hparams
        self.hp = hp
        self.hparams = hp
        self.peak_detection_params = defaultdict(
            lambda: {"prominence": 0.05, "width": None, "distance": None}
        )
        self.pr = defaultdict(
            lambda: {
                "train": PrecisionRecallMetric(),
                "val": PrecisionRecallMetric(),
                "test": PrecisionRecallMetric(),
            }
        )
        self.best_rval = defaultdict(
            lambda: {"train": (0, 0), "val": (0, 0), "test": (0, 0)}
        )
        self.overall_best_rval = 0
        self.stats = defaultdict(
            lambda: {"train": StatsMeter(), "val": StatsMeter(), "test": StatsMeter()}
        )

        wandb.init(
            project=self.hp.project, name=hp.exp_name, config=vars(hp), tags=[hp.tag]
        )
        self.build_model()

    def prepare_data(self):
        # setup training set
        if "timit" in self.hp.data:
            train, val, test = TrainTestDataset.get_datasets(path=self.hp.timit_path)
        elif "buckeye" in self.hp.data:
            train, val, test = TrainValTestDataset.get_datasets(
                path=self.hp.buckeye_path, percent=self.hp.buckeye_percent
            )
        else:
            raise Exception("no such training data!")

        if "libri" in self.hp.data:
            libri_train = LibriSpeechDataset(
                path=self.hp.libri_path,
                subset=self.hp.libri_subset,
                percent=self.hp.libri_percent,
            )
            train = ConcatDataset([train, libri_train])
            train.path = "\n\t+".join([dataset.path for dataset in train.datasets])
            print(f"added libri ({len(libri_train)} examples)")

        self.train_dataset = train
        self.valid_dataset = val
        self.test_dataset = test

        line()
        print("DATA:")
        print(f"train: {self.train_dataset.path} ({len(self.train_dataset)})")
        print(f"valid: {self.valid_dataset.path} ({len(self.valid_dataset)})")
        print(f"test: {self.test_dataset.path} ({len(self.test_dataset)})")
        line()

    @pl.data_loader
    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hp.batch_size,
            shuffle=True,
            collate_fn=collate_fn_padd,
            num_workers=self.hp.dataloader_n_workers,
        )
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padd,
            num_workers=self.hp.dataloader_n_workers,
        )
        return self.valid_loader

    @pl.data_loader
    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padd,
            num_workers=self.hp.dataloader_n_workers,
        )
        return self.test_loader

    def build_model(self):
        print("MODEL:")
        self.NFC = NextFrameClassifier(self.hp)
        line()

    def forward(self, data_batch, batch_i, mode):
        loss = 0

        # TRAIN
        audio, seg, phonemes, length, fname = data_batch
        preds = self.NFC(audio)
        NFC_loss = self.NFC.loss(preds, length)
        self.stats["nfc_loss"][mode].update(NFC_loss.item())
        loss += NFC_loss

        # INFERENCE
        if mode == "test" or (
            mode == "val" and self.hp.early_stop_metric == "val_max_rval"
        ):
            positives = 0
            for t in self.NFC.pred_steps:
                p = preds[t][0]
                p = replicate_first_k_frames(p, k=t, dim=1)
                positives += p
            positives = 1 - max_min_norm(positives)
            self.pr[f"cpc_{t}"][mode].update(seg, positives, length)

        loss_key = "loss" if mode == "train" else f"{mode}_loss"
        return OrderedDict({loss_key: loss})

    def generic_eval_end(self, outputs, mode):
        metrics = {}
        data = self.hp.data

        for k, v in self.stats.items():
            metrics[f"train_{k}"] = self.stats[k]["train"].get_stats()
            metrics[f"{mode}_{k}"] = self.stats[k][mode].get_stats()

        epoch = self.current_epoch + 1
        metrics["epoch"] = epoch
        metrics["current_lr"] = self.opt.param_groups[0]["lr"]

        line()
        for pred_type in self.pr.keys():
            if mode == "val":
                (precision, recall, f1, rval), (width, prominence, distance) = self.pr[
                    pred_type
                ][mode].get_stats()
                if rval > self.best_rval[pred_type][mode][0]:
                    self.best_rval[pred_type][mode] = rval, self.current_epoch
                    self.peak_detection_params[pred_type]["width"] = width
                    self.peak_detection_params[pred_type]["prominence"] = prominence
                    self.peak_detection_params[pred_type]["distance"] = distance
                    self.peak_detection_params[pred_type]["epoch"] = self.current_epoch
                    print(
                        f"saving for test - {pred_type} - {self.peak_detection_params[pred_type]}"
                    )
            else:
                print(
                        f"using pre-defined peak detection values - {pred_type} -"
                        f" {self.peak_detection_params[pred_type]}"
                )
                (precision, recall, f1, rval), _ = self.pr[pred_type][mode].get_stats(
                    width=self.peak_detection_params[pred_type]["width"],
                    prominence=self.peak_detection_params[pred_type]["prominence"],
                    distance=self.peak_detection_params[pred_type]["distance"],
                )
                # test has only one epoch so set it as best
                # this is to get the overall best pred_type later
                self.best_rval[pred_type][mode] = rval, self.current_epoch
            metrics[f"{data}_{mode}_{pred_type}_f1"] = f1
            metrics[f"{data}_{mode}_{pred_type}_precision"] = precision
            metrics[f"{data}_{mode}_{pred_type}_recall"] = recall
            metrics[f"{data}_{mode}_{pred_type}_rval"] = rval
            metrics[f"{data}_{mode}_{pred_type}_max_rval"] = self.best_rval[pred_type][
                mode
            ][0]
            metrics[f"{data}_{mode}_{pred_type}_max_rval_epoch"] = self.best_rval[
                pred_type
            ][mode][1]

        # get best rval from all rval types and all epochs
        best_overall_rval = -float("inf")
        for pred_type, rval in self.best_rval.items():
            if rval[mode][0] > best_overall_rval:
                best_overall_rval = rval[mode][0]
        metrics[f"{mode}_max_rval"] = best_overall_rval

        for k, v in metrics.items():
            print(f"\t{k:<30} -- {v}")
        line()
        wandb.log(metrics)

        output = OrderedDict({"log": metrics})

        return output

    def training_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, "train")

    def validation_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, "val")

    def test_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, "test")

    def validation_end(self, outputs):
        return self.generic_eval_end(outputs, "val")

    def test_end(self, outputs):
        return self.generic_eval_end(outputs, "test")

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optimizer == "sgd":
            self.opt = optim.SGD(
                parameters, lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4
            )
        elif self.hp.optimizer == "adam":
            self.opt = optim.Adam(parameters, lr=self.hparams.lr, weight_decay=5e-4)
        elif self.hp.optimizer == "ranger":
            self.opt = optim_extra.Ranger(
                parameters,
                lr=self.hparams.lr,
                alpha=0.5,
                k=6,
                N_sma_threshhold=5,
                betas=(0.95, 0.999),
                eps=1e-5,
                weight_decay=0,
            )
        else:
            raise Exception("unknown optimizer")
        print(f"optimizer: {self.opt}")
        line()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt, step_size=self.hp.lr_anneal_step, gamma=self.hp.lr_anneal_gamma
        )
        return [self.opt]

    def on_epoch_end(self):
        self.scheduler.step()

    def on_save_checkpoint(self, ckpt):
        ckpt["peak_detection_params"] = dill.dumps(self.peak_detection_params)

    def on_load_checkpoint(self, ckpt):
        self.peak_detection_params = dill.loads(ckpt["peak_detection_params"])

    def get_ckpt_path(self):
        return glob.glob(self.hp.wd + "/*.ckpt")[0]
