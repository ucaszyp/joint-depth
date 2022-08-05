import torch
import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor

class Testloader(torch.utils.data.Dataset):
    def __init__(self, test_dir, gt_dir):
        self._to_tensor = ToTensor()
        self.test_dir = test_dir
        self.gt_dir = gt_dir
        self.test_files = []
        self.gt_files = []

        test_files = sorted(os.listdir(self.test_dir))
        self.test_files += (test_files + test_files + test_files + test_files)
        
        # tmp = test_files[-1]
        # test_files.pop(-1)
        # test_files.insert(0, tmp)
        # self.test_files += test_files

        # tmp = test_files[-1]
        # test_files.pop(-1)
        # test_files.insert(0, tmp)
        # self.test_files += test_files

        # tmp = test_files[-1]
        # test_files.pop(-1)
        # test_files.insert(0, tmp)
        # self.test_files += test_files

        self.count = len(self.test_files)
        self.test_files = sorted(self.test_files)
        # for i in range(0, self.count, 4):
        #     print(self.test_files[i])
        
        self.gt_files = [os.path.join(self.gt_dir, (self.test_files[i]).split(".")[0] + ".npy") for i in range(self.count)]
        self.test_files = [os.path.join(self.test_dir, self.test_files[i]) for i in range(self.count)]
        

    def load_images(self, img_file, gt_file):
        img = cv2.imread(img_file)
        gt = np.load(gt_file)
        return img, gt
    
    def pack_data(self, img, gt):
        out = {}
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._to_tensor(img)
        img_gray = self._to_tensor(img_gray)
        gt = self._to_tensor(gt)
        out["color", 0, 0] = img
        out["color_gray", 0, 0] = img_gray
        out["gt", 0, 0] = gt

        return out

    def __getitem__(self, index):

        img, gt = self.load_images(self.test_files[index], self.gt_files[index])
        result = self.pack_data(img, gt)
        result["file_name"] = self.test_files[index]
        return result

    def __len__(self):
        return len(self.test_files)