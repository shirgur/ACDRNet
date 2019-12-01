import json
from glob import glob
import os
from tqdm import tqdm
import argparse
import itertools
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cityscapes instances data generator')
    parser.add_argument(
        '--outdir',
        type=str,
        default='/path/to/cityscapes_instances/',
        help='Output directory')
    parser.add_argument(
        '--outfile',
        type=str,
        default='all_classes_instances.json',
        help='Output file')
    parser.add_argument(
        '--annDir',
        type=str,
        default='/path/to/cityscapes_final_v5/',
        help='Annotations directory')
    parser.add_argument(
        '--min-area',
        type=float,
        default=100.0,
        help='Minimum Component Area')
    args = parser.parse_args()

    dataTypes = [('train_full', ['train', 'train_val']), 'train', 'train_val', 'val']

    for dataType in dataTypes:
        print('Processing - {}'.format(dataType))
        if isinstance(dataType, tuple):
            save_name = dataType[0]
            os.makedirs(os.path.join(args.outdir, save_name), exist_ok=True)
            all_files = []
            for subtype in dataType[1]:
                all_files += glob(os.path.join(args.annDir, subtype + '/**/*.json'))
        else:
            save_name = dataType
            os.makedirs(os.path.join(args.outdir, save_name), exist_ok=True)
            all_files = glob(os.path.join(args.annDir, dataType + '/**/*.json'))

        CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

        output_jason = dict()
        output_jason['classes_names'] = CLASSES
        output_jason['data'] = dict()
        for cls in CLASSES:
            output_jason['data'][cls] = []

        print('Extracting categories from {} files'.format(len(all_files)))
        for path in tqdm(all_files):
            json_data = json.load(open(path, 'r'))

            for instance in json_data:
                if instance['label'] in CLASSES:
                    data = dict()
                    data['labe'] = instance['label']
                    file_name = '/'.join(instance['img_path'].split('/')[-3:])
                    data['img'] = {'height': instance['img_height'],
                                   'width': instance['img_width'],
                                   'file_name': file_name}
                    segmentation = []
                    comp_bbox = []
                    for component in instance['components']:
                        if component['area'] > args.min_area:
                            segmentation.append(list(itertools.chain.from_iterable(component['poly'])))
                            comp_bbox.append(np.maximum(0, component['bbox']).tolist())
                    if not segmentation:
                        continue
                    data['segmentation'] = segmentation
                    data['comp_bbox'] = comp_bbox
                    data['bbox'] = np.maximum(0, instance['bbox']).tolist()
                    if 0 in data['bbox'][-2:]:
                        continue
                    output_jason['data'][instance['label']].append(data)

        print('Saving .json file')
        with open(os.path.join(args.outdir, save_name, args.outfile), 'w') as f:
            json.dump(output_jason, f)