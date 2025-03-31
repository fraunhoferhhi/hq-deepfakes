import zlib
import os 
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from torchvision import transforms

from ..conversion.utils.convutils import save_dict, setup_logger, load_masking_model

VIDEO_TYPES = [".mp4", ".avi"]
IMAGE_TYPES = [".png", ".jpg"]

### IMAGE
## Info Extractors
class InformationExtractor(ABC):
    '''
    Baseclass for information Extractors. These extractors are built to extract info such as landmarks or segmentation masks from a directory of images, which have been aligned by the face extractor.
    Extraction can be interrupted and continued. Note that the extracted information for all images will be stored in a dict (information_dict) and saved in a SINGLE FILE. 
    args:
        device: device to run networks on, cuda or cpu.
        batch_size: batch size for extraction networks. Set according to your machine.
        verbose: Whether to print minor information statements during the extraction process. For some reason statements are printed multiple times.
    '''
    def __init__(self,
                 device: str = "cuda",
                 batch_size: int = 8,
                 verbose: bool = True,
                 **kwargs):
        
        self.logger = setup_logger(verbose)

        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        self.__dict__.update(kwargs)

        # use this for saving later
        self.information_type = self.__class__.__name__.split("Extractor")[0].lower()
        self.information_dict = {}

        self._setup_model()


    @abstractmethod
    def _setup_model(self):
        # model needs to be set up differently depending on the type of information we want to extract
        ...

    def process(self,
                data_path: str,
                out_path: str = None):
        '''
        main function of the extractor, call this after init.
        args:
            data_path: path to the directory that holds the images that shall be processed.
            out_path: (Optional) where to store the extraced information (in a single file), if None its set to data_path        
        '''

        out_path = self._preprocess(data_path, out_path)

        image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f[-4:] in IMAGE_TYPES and f not in self.seen]
        image_files.sort()

        num_images = len(image_files)

        self.logger.info("Start processing of {} images".format(num_images))

        for i in range(0, num_images, self.batch_size):
            start = i 
            end = i + self.batch_size
            # obtain batch of images
            batch_files = image_files[start:end] if end < num_images else image_files[start:]
            batch = np.stack([self._process_image_file(f) for f in batch_files])
            # process batch (extract info and store in alignments dict)
            self._process_batch(batch, batch_files)

            save_dict(self.information_dict, os.path.join(out_path, "{}".format(self.information_type)))

            for img_path in batch_files:
                with open(self.seen_path, "a") as f:
                    f.write(img_path.rpartition("/")[-1])
                    f.write('\n')

        save_dict(self.information_dict, os.path.join(out_path, "{}".format(self.information_type)))

    def _preprocess(self, data_path: str, out_path: str = None):
        self.logger.info("Setting up {}".format(self.__class__.__name__))

        if out_path is not None:
            if not os.path.exists(out_path):
                self.logger.info("Creating out path at: {}".format(out_path))
                os.makedirs(out_path)
        else:
            out_path = data_path

        # handle seen path
        self.seen_path = os.path.join(data_path, "seen_{}.txt".format(self.information_type))
        if not os.path.exists(self.seen_path):
            f = open(self.seen_path, "x")
            seen = []
        else:
            f = open(self.seen_path, "r")
            seen = f.read()
            seen = seen.split('\n')
        self.seen = seen

        self.logger.info("Done!")

        return out_path

    def _process_batch(self, batch, batch_files):
        # batchwise preparation and extraction of information. then stores extracted info in dict.
        batch = self._prepare_batch(batch)
        info = self._detect_from_batch(batch)

        self._to_information_dict(info, batch_files)

    def _process_image_file(self, f):
        return cv2.imread(f)

    @abstractmethod
    def _prepare_batch(self, batch):
        # batches need to be prepared according to extractor model architecture.
        ...
    
    @abstractmethod
    def _detect_from_batch(self, batch):
        # forward pass through extraction network + some postprocessing
        ...

    def _to_information_dict(self, alignments, batch_files):
        # save extracted info to alignments dictionary based on the file name (so we can find correspondences later)
        for a, file in zip(alignments, batch_files):
            file_name = file.rpartition("/")[-1]

            self.information_dict[file_name] = a



class MasksExtractor(InformationExtractor):
    '''
    Extractor class to get segmentation masks from face images. For more info see InformationExtractor class comments.
    Note: this extractor should only be used with images that were extracted from the FaceExtractor class.
    '''
    def __init__(self,
                 device: str = "cuda",
                 batch_size: int = 8,
                 verbose: bool = True):
        
        super().__init__(device, batch_size, verbose)

    def _setup_model(self):
        # load masking model, see utils. The corresponding model state is also saved in this package.
        self.logger.info("Loading face segmentation network...")
        self.mask_net = load_masking_model(self.device)

    def _prepare_batch(self, batch):
        # to tensor / normalize
        self.logger.debug("Prep batch for masking")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        tensor = torch.zeros(batch.shape[0], 3, 512, 512)

        for i in range(batch.shape[0]):
            img = batch[i]
            img = cv2.resize(img, (512, 512), cv2.INTER_CUBIC)
            tensor[i] = transform(img)
        
        tensor = tensor.to(self.device)
        return tensor

    def _detect_from_batch(self, batch):
        # forward pass through masking network + compression of masks
        self.logger.debug("Compute batch face masks")
        # get masks
        with torch.no_grad():
            masks = self.mask_net(batch)[0].cpu().numpy().argmax(1)

        masks = masks.astype(np.uint8)
        masks = [zlib.compress(masks[n].tobytes()) for n in range(masks.shape[0])]

        return masks
