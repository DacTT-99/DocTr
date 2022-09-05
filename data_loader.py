import os

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from overrides import overrides
from torch.utils import data
from torch.utils.data import Dataset


class Doc3D_Seg(Dataset):
    def __init__(self, dataset_path: str = None, training: bool = True):
        super().__init__()
        self.dataset_path = dataset_path
        self.training = training

        self.samples = []
        self.process_sample()

    def process_sample(self):
        img_dir_path = os.path.join(self.dataset_path, "img")  # ./doc3d/img
        if not os.path.exists(img_dir_path):
            return
        for img_sub_dir in os.listdir(img_dir_path):
            img_sub_dir_path = os.path.join(img_dir_path, img_sub_dir)  # ./doc3d/img/1
            img_names = os.listdir(img_sub_dir_path)
            for img_name in img_names:
                fn = img_name.split(".")[0]
                img_path = os.path.join(self.dataset_path, "img", img_sub_dir, img_name)
                #bm_path = os.path.join(self.dataset_path, "bm", img_sub_dir, fn + ".mat")
                wc_path = os.path.join(self.dataset_path, "wc", img_sub_dir, fn + ".exr")
                if not (os.path.exists(img_path) and os.path.exists(wc_path)):
                    continue
                self.samples.append([img_path, wc_path])

    @overrides
    def __getitem__(self,index):
        img_path, wc_path = self.samples[index]
        img = cv2.imread(img_path)

        wc = cv2.imread(wc_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        wc = wc[:,:,0]
        wc[wc!=0] = 1.0      
        wc = torch.Tensor(wc)
        # bm = np.array(h5py.File(bm_path)['bm'])
        # bm = torch.Tensor(bm)
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
        return transform(img), wc

    def __len__(self):
        return len(self.samples)

class Doc3D_Rectify(Dataset):
    def __init__(self, dataset_path: str = None, training: bool = True):
        super().__init__()
        self.dataset_path = dataset_path
        self.training = training

        self.samples = []
        self.process_sample()

    def process_sample(self):
        img_dir_path = os.path.join(self.dataset_path, "img")  # ./doc3d/img
        if not os.path.exists(img_dir_path):
            return
        for img_sub_dir in os.listdir(img_dir_path):
            img_sub_dir_path = os.path.join(img_dir_path, img_sub_dir)  # ./doc3d/img/1
            img_names = os.listdir(img_sub_dir_path)
            for img_name in img_names:
                fn = img_name.split(".")[0]
                img_path = os.path.join(self.dataset_path, "img", img_sub_dir, img_name)
                bm_path = os.path.join(self.dataset_path, "bm", img_sub_dir, fn + ".mat")
                wc_path = os.path.join(self.dataset_path, "wc", img_sub_dir, fn + ".exr")
                if not(os.path.exists(img_path) and os.path.exists(img_path) and os.path.exists(img_path)):
                    continue
                self.samples.append([img_path,bm_path,wc_path])

    @overrides
    def __getitem__(self,index):
        img_path, bm_path ,wc_path= self.samples[index]
        img = cv2.imread(img_path)
        wc = cv2.imread(wc_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img[wc==0] = 0
        img = cv2.resize(img,(288,288))

        bm = np.array(h5py.File(bm_path)['bm'])
        bm = np.reshape(bm,(448,448,2))
        bm = cv2.resize(bm,(288,288))
        bm = np.reshape(bm,(2,288,288))
        bm = torch.Tensor(bm)

        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
        return transform(img), bm

    def __len__(self):
        return len(self.samples)

if __name__ =='__main__':
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    dataset = Doc3D_Seg('/home/list_99/data/doc3D')
    print(len(dataset))
    #train_loader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=True)
    a,b=dataset.__getitem__(18)