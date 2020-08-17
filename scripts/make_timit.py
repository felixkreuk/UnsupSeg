
import argparse
import os
import shutil
from tqdm import tqdm

def main(inpath, outpath):
    if not os.path.exists(inpath): 
        print('Error: input path does not exist!!')
        return -1 
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    for _f in tqdm(os.listdir(inpath)):
        parent_f = os.path.join(inpath, _f)
        if os.path.isdir(parent_f):
            for _ff in os.listdir(parent_f):
                parent_ff = os.path.join(parent_f, _ff)
                if os.path.isdir(parent_ff):
                    for ex in os.listdir(parent_ff):
                        if ex.endswith('.phn') or ex.endswith('.wav'): 
                            src_name = os.path.join(parent_ff, ex)
                            tgt_name = os.path.join(outpath, _f+'_'+_ff+'_'+ex)
                            shutil.copy(src_name, tgt_name)

parser = argparse.ArgumentParser(description='Make TIMIT dataset ready for unsupervised segmentation.')
parser.add_argument('--inpath', type=str, required=True, help='the path to the base timit dir.')
parser.add_argument('--outpath', type=str, required=True, help='the path to save the new format of the data.')

args = parser.parse_args()

main(args.inpath, args.outpath)

