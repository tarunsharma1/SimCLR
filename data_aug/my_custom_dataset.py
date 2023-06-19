'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import csv
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator



class CTDataset(Dataset):

    def __init__(self, data_root, split, transform):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = data_root
        self.split = split
        self.transform = transform
        # self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        #     Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
        #     ToTensor()                          # ...and convert them to torch.Tensor.
        # ])
        
        # index data into list
        self.data = []

        self.label_mapping = {}
        global_mapping_idx = 0

        # if split == 'train':
        #     f = open(cfg['train_label_file'], 'r')
        # elif split=='val':
        #     f = open(cfg['val_label_file'], 'r')
        # elif split=='test':
        #     f = open(cfg['test_label_file'], 'r')
        if split=='unlabeled':
            f = open('/home/tsharma/tarun_code/5_percent_labels.csv', 'r')
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            
            self.data.append([row[0]])

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name = self.data[idx][0]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor