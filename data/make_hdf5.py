import h5py
import os
from glob import glob
import numpy as np
from imageio import imread
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Convert Cityscapes to HDF5')
parser.add_argument('--images-path', type=str,
                    default='/path/to/leftImg8bit',
                    help='Path to <leftImg8bit> Cityscapes dataset')
parser.add_argument('--outdir', type=str,
                    default='/path/to/cityscapes_instances',
                    help='Save path (should be the same as the output path of generate_cityscapes_instances.py')
args = parser.parse_args()

images_path = glob(os.path.join(args.images_path, '**/**/*.png'))
short_path = [s[55:] for s in images_path]
file_name = 'all_images.hdf5'

with h5py.File(os.path.join(args.outdir, file_name), 'a') as f:
    for i, path in enumerate(tqdm(images_path)):
        image = imread(path)
        data_seg = f.create_dataset(short_path[i], image.shape, data=image, compression="gzip", dtype=np.uint8)