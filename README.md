# Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation (INTERSPEECH 2020)

## Paper
[Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation](https://arxiv.org/abs/2007.13465).
</br>
Felix Kreuk, Joseph Keshet, Yossi Adi
</br>
INTERSPEECH 2020

We propose a self-supervised representation learning model for the task of unsupervised phoneme boundary detection. The model is a convolutional neural network that operates directly on the raw waveform. It is optimized to identify spectral changes in the signal using the Noise-Contrastive Estimation principle. At test time, a peak detection algorithm is applied over the model outputs to produce the final boundaries. As such, the proposed model is trained in a fully unsupervised manner with no manual annotations in the form of target boundaries nor phonetic transcriptions. We compare the proposed approach to several unsupervised baselines using both TIMIT and Buckeye corpora. Results suggest that our approach surpasses the baseline models and reaches state-of-the-art performance on both data sets. Furthermore, we experimented with expanding the training set with additional examples from the Librispeech corpus. We evaluated the resulting model on distributions and languages that were not seen during the training phase (English, Hebrew and German) and showed that utilizing additional untranscribed data is beneficial for model performance.

If you find this paper and implementation useful, please consider citing our work:
```
@article{kreuk2020self,
  title={Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation},
  author={Kreuk, Felix and Keshet, Joseph and Adi, Yossi},
  journal={arXiv preprint arXiv:2007.13465},
  year={2020}
}
```

## Clone repository
```
git clone https://github.com/felixkreuk/UnsupSeg.git
cd UnsupSeg
```

## Setup environment
```
conda create --name unsup_seg --file requirements.txt
conda activate unsup_seg
```

## Data structure
The training script assumes that the data is structured as follows:
```
timit_directory
│
└───val
│   │   X.wav
│   └─  X.phn
│
└───test
│   │   Y.wav
│   └─  Y.phn
│
└───train
    │   Z.wav
    └─  Z.phn
```

Where `X.wav` is a raw waveform signal, and `X.phn` is its' corresponding phoneme boundaries labeld with the following format:
```
0 9640 h#
9640 11240 sh
11240 12783 iy
12783 14078 hv
14078 16157 ae
16157 16880 dcl
...
```
Where the two numbers each line represent the onset of offset of the phoneme (in samples), and the last element represents the phoneme identity.

## Usage
### Train
To run training with default hyper-parameters, run the following:
```
python main.py
```
To see further hyper-parameters see `conf/config.yaml`.
More examples:
```
python main.py train=timit  # train on timit
python main.py train=buckeye  # train on buckeye
python main.py train=timit_libri  # timit + librispeech
python main.py train=timit_libri libri_percent=0.5  # use only 50% of librispeech
```

### Test
The following command runs a test epoch on the selected data and reports results in terms of precision, recall, F1 and R-value.
```
python main.py ckpt=/absolute/path/to/model.ckpt train=timit  # test on timit
python main.py ckpt=/absolute/path/to/model.ckpt train=buckeye  # test on buckeye
```

### Inference on a single wav
The following command runs inference on a single .wav file and outputs the predicted boundaries in seconds.
```
python predict.py --ckpt /absolute/path/to/model.ckpt --wav /path/to/audio.wav
python predict.py --ckpt /absolute/path/to/model.ckpt --wav /path/to/audio.wav --prominence 0.05
```
The threshold for the peak detection procedure can be adjusted using the `--prominence 0.05` argument. For best results it is advisable to trim silences using a voice activity detector.
