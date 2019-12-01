from torch.utils.data import Dataset
import json
import pycocotools.mask as maskUtils
import numpy as np
import h5py
import os


class CityscapesInstances(Dataset):
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def __init__(self,
                 data_inst_path,
                 ann_file,
                 class_name,
                 transformations=None,
                 loops=100):
        super(CityscapesInstances, self).__init__()
        self.data_inst_path = data_inst_path
        self.ann_file = ann_file
        self.class_name = class_name
        self.transformations = transformations
        self.loops = loops

        print('loading \"{}\" annotations into memory...'.format(ann_file))
        self.data = json.load(open(os.path.join(data_inst_path, ann_file, 'all_classes_instances.json'), 'r'))
        self.images = None

        self.data_length = len(self.data['data'][self.class_name])

        self.calsses = self.data['classes_names']

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def __len__(self):
        return self.data_length * self.loops

    def __getitem__(self, item):
        item = item % self.data_length
        if self.images is None:
            self.images = h5py.File(os.path.join(self.data_inst_path, 'all_images.hdf5'), 'r')
        ann = self.data['data'][self.class_name][item]
        mask = self._poly2mask(ann['segmentation'], ann['img']['height'], ann['img']['width'])
        bbox = np.maximum(0, np.array(ann['bbox']).astype(np.int32))

        instance_image = self.images[ann['img']['file_name']][bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        instance_mask = mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

        if self.transformations is not None:
            instance_image, instance_mask = self.transformations(instance_image, instance_mask)

        return instance_image, instance_mask


class CityscapesInstances_comp(Dataset):
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def __init__(self,
                 data_inst_path,
                 ann_file,
                 class_name,
                 transformations=None,
                 loops=100):
        super(CityscapesInstances_comp, self).__init__()
        self.data_inst_path = data_inst_path
        self.ann_file = ann_file
        self.class_name = class_name
        self.transformations = transformations
        self.loops = loops

        print('loading \"{}\" annotations into memory...'.format(ann_file))
        self.data = json.load(open(os.path.join(data_inst_path, ann_file, 'all_classes_instances.json'), 'r'))
        self.images = None

        self.data_length = len(self.data['data'][self.class_name])

        self.calsses = self.data['classes_names']
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def __len__(self):
        return self.data_length * self.loops

    def __getitem__(self, item):
        item = item % self.data_length
        if self.images is None:
            self.images = h5py.File(os.path.join(self.data_inst_path, 'all_images.hdf5'), 'r')
        ann = self.data['data'][self.class_name][item]
        polygons = ann['segmentation']
        comp_bbox = ann['comp_bbox']
        i = np.random.randint(0, len(polygons))
        p = polygons[i]
        bb = comp_bbox[i]
        mask = self._poly2mask([p], ann['img']['height'], ann['img']['width'])
        bbox = np.array(bb).astype(np.int32)

        instance_image = self.images[ann['img']['file_name']][bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        instance_mask = mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

        if self.transformations is not None:
            instance_image, instance_mask = self.transformations(instance_image, instance_mask)

        return instance_image, instance_mask
