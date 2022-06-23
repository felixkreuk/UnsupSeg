import random
import soundfile as sf
import buckeye
from tqdm import tqdm
import numpy as np
from boltons import fileutils
import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--spkr', default=False, action='store_true')
parser.add_argument('--source', default=False)
parser.add_argument('--target', default=False)
parser.add_argument('--min_phonemes', type=int)
parser.add_argument('--max_phonemes', type=int)
args = parser.parse_args()


DELIMITER = ['VOCNOISE', 'NOISE', 'SIL']
FORBIDDEN = ['{B_TRANS}', '{E_TRANS}', '<EXCLUDE-name>', 'LAUGH', 'UNKNOWN', 'IVER-LAUGH', '<exclude-Name>', 'IVER']
MIN_PHONEMES = args.min_phonemes
MAX_PHONEMES = args.max_phonemes
NOISE_EDGES = 0.2
is_delim = lambda x: x.seg in DELIMITER
contain_forbidden = lambda phone_list: not set([p.seg for p in phone_list]).isdisjoint(FORBIDDEN)
path = args.source
output_path = args.target
train_path = osp.join(output_path, "train")
val_path = osp.join(output_path, "val")
test_path = osp.join(output_path, "test")

wavs = list(fileutils.iter_find_files(path, "*.wav"))
files = []
segments = []
file_counter = 0

os.makedirs(output_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for wav in tqdm(wavs):
    try:
        spkr   = osp.basename(wav)[:3]
        name   = wav.replace(".wav", "")
        words  = wav.replace("wav", "words")
        phones = wav.replace("wav", "phones")
        log    = wav.replace("wav", "log")
        txt    = wav.replace("wav", "txt")
        track  = buckeye.Track(name=name,
                               words=words,
                               phones=phones,
                               log=log,
                               txt=txt,
                               wav=wav)
        phones = track.phones[1:-1]
        delim_locations = np.array([i for i, phone in enumerate(phones) if is_delim(phone)])
        loaded_wav, sr = sf.read(wav)

        # in some files the last segment annotation ends after the
        # actual wav file, ignore those files.
        if phones[-1].end >= loaded_wav.shape[0] / sr:
            print(f"last phone end: {phones[-1].end}")
            print(f"len of wav: {loaded_wav.shape[0] / sr}")
            print(f"skipping {wav}")
            continue

        # iterate over all phone segments inside wav
        for start, end in zip(delim_locations[:-1], delim_locations[1:]):
            segment = phones[start:end+1]

            # if contains forbidden annotations, or number of segments is
            # not in the desired range => ignore
            if contain_forbidden(segment) or not (MIN_PHONEMES < end - start < MAX_PHONEMES):
                continue

            # make sure that the noise/sil on the edges is less than
            # NOISE_EDGES seconds
            if segment[0].end - segment[0].beg > NOISE_EDGES:
                segment[0]._beg = segment[0].end - NOISE_EDGES
            if segment[-1].end - segment[-1].beg > NOISE_EDGES:
                segment[-1]._end = segment[-1].beg + NOISE_EDGES

            # get stat and end times
            segment_start_time = segment[0].beg
            segment_end_time   = segment[-1].end

            # trim wav according to start and end
            # also, extract from the .phn file the corresponding phonemes
            output_wav_file = f"{spkr}_{file_counter}.wav"
            output_phn_file = f"{spkr}_{file_counter}.phn"
            track.clip_wav(osp.join(output_path, output_wav_file), segment_start_time, segment_end_time)
            phn_data = "\n".join([f"{int((p.beg - segment_start_time) * sr)} {int((p.end - segment_start_time) * sr)} {p.seg}" for p in segment])
            with open(osp.join(output_path, output_phn_file), "w") as f:
                f.writelines(phn_data)
            file_counter += 1

            segments.append(segment)
    except UnboundLocalError:
        print(f"loading {wav} failed!")

lens = np.array([len(seg) for seg in segments])
secs = np.array([seg[-1].end - seg[0].beg for seg in segments])
print(f"{len(segments)} items")
print(f"avg len: {lens.mean()}")
print(f"min len: {lens.min()}")
print(f"max len: {lens.max()}")
print(f"avg sec: {secs.mean()}")
print(f"min sec: {secs.min()}")
print(f"max sec: {secs.max()}")
print(f"{secs.sum() / (60*60)} hours")

if args.spkr:
    os.chdir(output_path)
    test_spkrs = ["s07", "s03", "s31", "s34"]
    val_spkrs  = ["s40", "s39", "s36", "s25"]
    for spkr in test_spkrs:
        os.system(f"mv {spkr}* test/")
    for spkr in val_spkrs:
        os.system(f"mv {spkr}* val/")
    os.system(f"mv *.wav train/")
    os.system(f"mv *.phn train/")
else:
    splits = [0.8, 0.9, 1.0]
    wavs = list(fileutils.iter_find_files(output_path, "*.wav"))
    random.shuffle(wavs)
    for i, wav in enumerate(wavs):
        if i < len(wavs) * splits[0]:
            target = train_path
        elif len(wavs) * splits[0] <= i and i < len(wavs) * splits[1]:
            target = val_path
        else:
            target = test_path
        phn = wav.replace("wav", "phn")
        os.rename(wav, wav.replace(output_path, target + "/"))
        os.rename(phn, phn.replace(output_path, target + "/"))
