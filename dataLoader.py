
# from cv2 import imread 
import cv2
import numpy as np
from random import randint
from torch.autograd import Variable

# from torchvision import datasets
from torch.utils.data import DataLoader

# from PIL import Image

from torch.utils.data.dataset import Dataset

import os
import pathlib
from collections import defaultdict
from random import shuffle
import pdb
import util

# training data loader
class TrainingSetLoader(Dataset):
    def __init__(self, root, transforms=None):
        """ Args:
            root (string): path to img file folder
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.root = root
        self.images = self.retrieve_images()
        self.transforms = transforms

    def __getitem__(self, index):
        # get the images based on the index 
        item = self.images[index]
        # Return image matrix and the corresponding classification
        return item['mat'], item['val']

    def __len__(self):
        return len(self.images)

    # retrieve every image
    def retrieve_images(self):
        # # remove .DS_Store if it exist
        # if '.DS_Store' in phone_files:
        #     phone_files.remove('.DS_Store')
        # if '.DS_Store' in background_files:
        #     background_files.remove('.DS_Store')
        images,labels,_ = util.readDataset(self.root)
        img_ph_list,img_bg_list = util.cropOut(images,labels)
        ph_dict = [{'mat': np.rollaxis(x, 2, 0) , 'val': 1} for x in img_ph_list]
        bg_dict = [{'mat': np.rollaxis(x, 2, 0) , 'val': 0} for x in img_bg_list]
        mix_dict = ph_dict + bg_dict
        shuffle(mix_dict)
        return mix_dict

class InferingSetLoader(Dataset):
    def __init__(self, root):
        """ Args:
            root (string): path to img file folder
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.root = root
        self.images = self.retrieve_images()
        # self.transforms = transforms

    def __getitem__(self, index):
        # get the images based on the index 
        item = self.images[index]
        # Return image matrix and the corresponding classification
        return item['mat']

    def __len__(self):
        return len(self.images)

    # retrieve every image
    def retrieve_images(self):
        # # remove .DS_Store if it exist
        # if '.DS_Store' in phone_files:
        #     phone_files.remove('.DS_Store')
        # if '.DS_Store' in background_files:
        #     background_files.remove('.DS_Store')
        # images,labels,_ = util.readDataset(self.root)
        win_list = util.slideWindow(self.root)
        win_list = [{'mat': np.rollaxis(x, 2, 0)} for x in img_ph_list]
        return win_list