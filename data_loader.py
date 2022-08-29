import os

import torch
from overrides import overrides
from torch.utils.data import Dataset


class Doc3D(Dataset):
    def __init__(self,
                 dataset_path: str = None,
                 training: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.training = training
        self.process_sample()

    def process_sample(self):
        img_dir_paths = os.listdir(os.path.join(self.dataset_path,'img'))   #home/list_99/Download/doc3d/img
        bm_dir_paths = os.listdir(os.path.join(self.dataset_path,'bm'))     #home/list_99/Download/doc3d/bm
        imgs_path = []
        bms_path = []
        sample = []
        for dir_path in img_dir_paths:
            img_sub_dir += os.listdir(os.path.join(self.dataset_path,dir_path)) # 1,2,3,4,5 ... 21
            for img_path in imgs_path:
                img_sub_dir_path = os.path.join(self.dataset_path,dir_path,img_sub_dir)           #home/list_99/Download/doc3d/img/1
                bm_sub_dir_path = os.path.join(self.dataset_path,dir_path,img_sub_dir)           #home/list_99/Download/doc3d/bm/1
        self.check_valid_sample(imgs_path)

    def check_valid_sample(self, imgs_path: list):
        for img_path in imgs_path:
            bm_path = img.path

    @overrides
    def __getitem__():
        pass
