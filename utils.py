import random
import torch
import torch.nn as nn
import numpy as np
import time
from scipy.signal import find_peaks
import wandb
from tqdm import tqdm


def replicate_first_k_frames(x, k, dim):
    return torch.cat([x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device)), x], dim=dim)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class PrintShapeLayer(nn.Module):
    def __init__(self):
        super(PrintShapeLayer, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def detect_peaks(x, lengths, prominence=0.1, width=None, distance=None):
    """detect peaks of next_frame_classifier
    
    Arguments:
        x {Tensor} -- batch of confidence per time
    """ 
    out = []

    for xi, li in zip(x, lengths):
        if type(xi) == torch.Tensor:
            xi = xi.cpu().detach().numpy()
        xi = xi[:li]  # shorten to actual length
        xmin, xmax = xi.min(), xi.max()
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)

        if len(peaks) == 0:
            peaks = np.array([len(xi)-1])

        out.append(peaks)

    return out


class PrecisionRecallMetric:
    def __init__(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.eps = 1e-5
        self.data = []
        self.tolerance = 2
        self.prominence_range = np.arange(0, 0.15, 0.01)
        self.width_range = [None, 1]
        self.distance_range = [None, 1]

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        EPS = 1e-7
        
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval

    def zero(self):
        self.data = []

    def update(self, seg, pos_pred, length):
        for seg_i, pos_pred_i, length_i in zip(seg, pos_pred, length):
            self.data.append((seg_i, pos_pred_i.cpu().detach().numpy(), length_i.item()))

    def get_stats(self, width=None, prominence=None, distance=None):
        print(f"calculating metrics using {len(self.data)} entries")
        max_rval = -float("inf")
        best_params = None
        segs = list(map(lambda x: x[0], self.data))
        length = list(map(lambda x: x[2], self.data))
        yhats = list(map(lambda x: x[1], self.data))

        width_range = self.width_range
        distance_range = self.distance_range
        prominence_range = self.prominence_range

        # when testing, we would override the search with specific values from validation
        if prominence is not None:
            width_range = [width]
            distance_range = [distance]
            prominence_range = [prominence]

        for width in width_range:
            for prominence in prominence_range:
                for distance in distance_range:
                    precision_counter = 0
                    recall_counter = 0
                    pred_counter = 0
                    gt_counter = 0
                    peaks = detect_peaks(yhats,
                                         length,
                                         prominence=prominence,
                                         width=width,
                                         distance=distance)

                    for (y, yhat) in zip(segs, peaks):
                        for yhat_i in yhat:
                            min_dist = np.abs(y - yhat_i).min()
                            precision_counter += (min_dist <= self.tolerance)
                        for y_i in y:
                            min_dist = np.abs(yhat - y_i).min()
                            recall_counter += (min_dist <= self.tolerance)
                        pred_counter += len(yhat)
                        gt_counter += len(y)

                    p, r, f1, rval = self.get_metrics(precision_counter,
                                                      recall_counter,
                                                      pred_counter,
                                                      gt_counter)
                    if rval > max_rval:
                        max_rval = rval
                        best_params = width, prominence, distance
                        out = (p, r, f1, rval)
        self.zero()
        print(f"best peak detection params: {best_params} (width, prominence, distance)")
        return out, best_params


class StatsMeter:
    def __init__(self):
        self.data = []

    def update(self, item):
        if type(item) == list:
            self.data.extend(item)
        else:
            self.data.append(item)

    def get_stats(self):
        data = np.array(self.data)
        mean = data.mean()
        self.zero()
        return mean

    def zero(self):
        self.data.clear()
        assert len(self.data) == 0, "StatsMeter didn't clear"


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.msg} -- started")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"{self.msg} -- done in {(time.time() - self.start_time)} secs")


def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x


def line():
    print(90 * "-")
