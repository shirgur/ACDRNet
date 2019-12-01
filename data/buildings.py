from torch.utils.data import Dataset
from imageio import imread
import os


class BuildingsDataset(Dataset):
    def __init__(self,
                 data_path,
                 ann='train',
                 transformations=None,
                 train_size=100,
                 test_size=68):
        self.img_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.transformations = transformations
        self.ann = ann

        self.img_list = os.listdir(self.img_path)
        self.img_list.sort()
        self.mask_list = os.listdir(self.mask_path)
        self.mask_list.sort()

        if self.ann == 'train':
            self.img_list = self.img_list[0:train_size]
            self.mask_list = self.mask_list[0:train_size]
        else:
            self.img_list = self.img_list[train_size:train_size + test_size]
            self.mask_list = self.mask_list[train_size:train_size + test_size]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_file = self.img_list[item]
        mask_file = self.mask_list[item]

        image = imread(self.img_path + '/' + img_file)
        mask = imread(self.mask_path + '/' + mask_file)

        if self.transformations is not None:
            image, mask = self.transformations(image, mask)

        return image, mask
