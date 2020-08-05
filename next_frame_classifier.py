
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from utils import LambdaLayer, PrintShapeLayer, length_to_mask
from dataloader import TrainTestDataset
from collections import defaultdict


class NextFrameClassifier(nn.Module):
    def __init__(self, hp):
        super(NextFrameClassifier, self).__init__()
        self.hp = hp

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.enc = nn.Sequential(
            nn.Conv1d(1, LS, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, Z_DIM, kernel_size=4, stride=2, padding=0, bias=False),
            LambdaLayer(lambda x: x.transpose(1,2)),
        )
        print("learning features from raw wav")
        
        if self.hp.z_proj != 0:
            if self.hp.z_proj_linear:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
            else:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, Z_DIM), nn.LeakyReLU(),
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
                
        # # similarity estimation projections
        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")

    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef
    
    def forward(self, spect):
        device = spect.device

        # wav => latent z
        z = self.enc(spect.unsqueeze(1))
        
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            pos_pred = self.score(z[:, :-t], z[:, t:])  # score for positive frame
            preds[t].append(pos_pred)

            for _ in range(self.hp.n_negatives):
                if self.training:
                    time_reorder = torch.randperm(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    if self.hp.batch_shuffle:
                        batch_reorder = torch.randperm(pos_pred.shape[0])
                else:
                    time_reorder = torch.arange(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    
                neg_pred = self.score(z[:, :-t], z[batch_reorder][: , time_reorder])  # score for negative random frame
                preds[t].append(neg_pred)
            
        return preds

    def loss(self, preds, lengths):
        loss = 0
        for t, t_preds in preds.items():
            mask = length_to_mask(lengths - t)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[...,0] * mask
            loss += -out.mean()
        return loss

@hydra.main(config_path='conf/config.yaml', strict=False)
def main(cfg):
    ds, _, _ = TrainTestDataset.get_datasets(cfg.timit_path)
    spect, seg, phonemes, length, fname = ds[0]
    spect = spect.unsqueeze(0)

    model = NextFrameClassifier(cfg)
    out = model(spect, length)


if __name__ == "__main__":
    main()
