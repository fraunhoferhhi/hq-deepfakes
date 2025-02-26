import os
import zlib
import pickle

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import Augmentator

class DeepfakeDataset(Dataset):
    '''
    Specific Dataset for training of Deepfake Autoencoders wiht Pytorch Lightning. Performs blurring and roi cropping here instead of in the loss.
    Loads images, generates corresponding targets and calls augmentation.
    Also loads the face, mouth and eye masks. Returns data for both sides a and b simultaneosly! Do not use this set like other standard Pytorch datasets, it wont work.
    params:
        path: path to images of person 
        input_size: dim of images in the faceset
        output_size: dim of augmented images (inputs for the model)
        coverage_ratio: amount of image to use (center crop)
        no_flip: Indicates wheter random flipping occurs during augmentation
    '''
    def __init__(self, 
                 path: str, 
                 input_size: int = 512, 
                 model_img_size: int = 256, 
                 coverage_ratio: float=  0.8, 
                 no_flip: bool = False):
        
        super().__init__()

        # save attributes
        self.path = path
        self.input_size = input_size
        self.model_input_size = model_img_size
        self.model_output_size = model_img_size
        self.coverage_ratio = coverage_ratio
        self.no_flip = no_flip

        # setup
        self._setup()

    def _setup(self):
        self.augmentator = Augmentator(self.input_size, 
                                       self.model_input_size, 
                                       self.model_output_size,
                                       self.coverage_ratio, 
                                       self.no_flip)

        self.files = [f for f in os.listdir(self.path) if f.endswith(".png")]

        # load alignments
        with open(os.path.join(self.path, "masks.fsa"), "rb") as s:
            self.alignments_a = s.read()
        self.alignments_a = pickle.loads(zlib.decompress(self.alignments_a))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tt = transforms.ToTensor()

        X_a = cv2.imread(os.path.join(self.path, self.files[idx]))

        mask_a = self.alignments_a[self.files[idx]]
        mask_a = np.frombuffer(zlib.decompress(mask_a), dtype="uint8").reshape((self.input_size, self.input_size))
        mask_a = self._clean_mask(mask_a)

        X_a, y_a, mask_a = self.augmentator.process(X_a, mask_a)

        # transform to tensor
        X_a = tt(X_a)

        y_a = tt(y_a)

        mask_a = torch.from_numpy(mask_a)


        return X_a, y_a, mask_a
    
    def _clean_mask(self, mask):
        face_points = np.logical_or(np.logical_and(mask<7, mask>0), np.logical_and(mask>9, mask<14)).astype(np.uint8)

        mouth_points = np.logical_or(mask==11, np.logical_or(mask==12, mask==13)).astype(np.uint8)

        eye_points = (mask == 4)*1.0 + (mask == 5)*1.0
        eye_points = eye_points.astype(np.uint8)

        if len(np.unique(eye_points)) == 1:
            eye_points = (mask==6)*1.0
            eye_points = eye_points.astype(np.uint8)

        mask = face_points + eye_points*9 + mouth_points*19

        return mask
    
