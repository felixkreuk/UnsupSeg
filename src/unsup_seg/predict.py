import argparse
import dill
from argparse import Namespace
import librosa
import torch
import torchaudio
from utils import detect_peaks, max_min_norm, replicate_first_k_frames
from next_frame_classifier import NextFrameClassifier


SR = 16000


def main(wav, ckpt, prominence):
    print(f"running inference on: {wav}")
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))

    # load weights and peak detection params
    model = NextFrameClassifier(hp)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    peak_detection_params = dill.loads(ckpt["peak_detection_params"])["cpc_1"]
    if prominence is not None:
        print(f"overriding prominence with {prominence}")
        peak_detection_params["prominence"] = prominence

    # load data
    signal, sr = librosa.core.load(wav, sr=SR)
    assert (
        sr == 16000
    ), "model was trained with audio sampled at 16khz, please downsample."
    audio = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    # run inference
    preds = model(audio)  # get scores
    preds = preds[1][0]  # get scores of positive pairs
    preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
    preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)
    preds = detect_peaks(
        x=preds,
        lengths=[preds.shape[1]],
        prominence=peak_detection_params["prominence"],
        width=peak_detection_params["width"],
        distance=peak_detection_params["distance"],
    )  # run peak detection on scores
    preds = preds[0] * 160 / sr  # transform frame indexes to seconds

    print("predicted boundaries (in seconds):")
    print(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unsupervised segmentation inference script"
    )
    parser.add_argument("--wav", help="path to wav file")
    parser.add_argument("--ckpt", help="path to checkpoint file")
    parser.add_argument(
        "--prominence",
        type=float,
        default=None,
        help="prominence for peak detection (default: 0.05)",
    )
    args = parser.parse_args()
    main(args.wav, args.ckpt, args.prominence)
