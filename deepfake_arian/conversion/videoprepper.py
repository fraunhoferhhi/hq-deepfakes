import os
import zlib

import cv2
import numpy as np

import face_alignment
import torch
from torchvision import transforms

from tqdm import tqdm

from faceex_arian.utils.utils import load_masking_model, load_video, setup_logger, save_dict, PoseEstimate, _MEAN_FACE, umeyama

class VideoPrepper():
    '''
    VideoPrepper class. Use this to extract relevant information from a video before performing deepfake conversion. Can be called on a single video or a directory (all videos in it)
    args:
        device: where to run networks on 
        batch_size: batch_size of information extractor networks
        verbose: whether to log minor information in the process
    '''
    def __init__(self,
                 device: str = "cuda",
                 batch_size: int = 8,
                 verbose: bool = True):

        self.logger = setup_logger(verbose)

        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        self._setup_networks()

    def _setup_networks(self):
        # Load face aligner and masking network. Install faceex_arian package before running this.
        self.logger.info("Loading alignment and mask networks")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)
        self.mask_net = load_masking_model(self.device)

    def process_dir(self, dir):
        # call process video on function on the path specified in dir argument.
        self.logger.info("Processing directory: {}".format(dir))

        videos = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("mp4") or f.endswith("avi")]
        for v in videos:
            self.process_video(v)


    def process_video(self,
                      video_path: str):
        # extract rotation matrices and (aligned) masks from given video
        self.logger.info("Processing video: {}".format(video_path))
        video = load_video(video_path, 1, self.verbose)
        video_name = video_path.split("/")[-1].split(".")[0]

        video_alignments = {}

        num_frames = video.shape[0]
        with torch.no_grad():
            for i in tqdm(range(0, num_frames, self.batch_size)):
                start = i
                end = i+self.batch_size if i+self.batch_size < num_frames else num_frames
                indices = list(range(start, end))

                batch = video[start:end]
                # detect landmarks, need those for alignment and rotation matrices
                landmarks, batch, indices = self._detect_from_batch(batch, indices)

                if len(landmarks) == 0:
                    continue

                # obtain rotation matrices and align faces for mask extraction
                batch, matrices = self._align_batch(batch, landmarks)
                # mask extraction
                masks = self._get_masks(batch)

                # save masks and rotation matrices in video_alignments
                video_alignments = self._update_dict(video_alignments, matrices, masks, indices, video_name)

        # save alignments in prep file
        alignments_path = video_path.split(".")[0] + "_prep"
        save_dict(video_alignments, alignments_path, file_type="fsa")

    # saving
    def _update_dict(self, alignments, matrices, masks, indices, name):
        # save rotation matrices and masks in alignments dict
        size = len(masks)
        self.logger.debug("Writing to dict")

        for frame in range(size):
            index = indices[frame]
            m = masks[frame]
            mat = matrices[frame]

            for (mask, matrix) in zip(m, mat):
                suffix = "_{}.png".format(index)
                key = name + suffix
                alignments[key] = {
                    "mask": zlib.compress(mask.tobytes()),
                    "matrix": matrix,
                }


        return alignments

    # for matrices / alignment
    def _prep_batch_for_detect(self, batch):
        # transform batch to tensor in order to compute landmarks
        self.logger.debug("Prep batch for detection")

        size, h, w = batch.shape[:3]
        tensor = torch.zeros(size, 3, h, w)

        for i in range(size):
            img = np.moveaxis(batch[i], -1 , 0)
            tensor[i] = torch.from_numpy(img)

        return tensor

    def _detect_from_batch(self, batch, indices):
        # forward pass through face alignment network, only use faces of specific size and only first face in video
        self.logger.debug("Detecting landmarks")

        batch_copy = batch.copy()

        batch = self._prep_batch_for_detect(batch)
        batch = batch.to(self.device)

        with torch.no_grad():
            landmarks, scores, boxes = self.fa.get_landmarks_from_batch(batch, return_bboxes=True, return_landmark_score=True)

        # clear a bit of vram
        batch = None

        index = []
        # clean from false positives / small faces
        for i in range(len(landmarks)):
            if scores[i] is None:
                # this handles the case if no face is detected / to be seen
                index.append(False)
            else:
                pos = np.argmax(np.mean(scores[i], axis=1))
                box = boxes[i][pos]
                if (box[2] - box[0] > 95) or (box[3] - box[1] > 125):
                    # if the largest face is large enough the frame is taken otherwise discarded, should recode this based on scores
                    index.append(True)
                else:
                    index.append(False)

        # get valid landmarks, images and indices (latter are only for saving with the right frame index)
        indices_final = [indices[i] for i in range(len(indices)) if index[i]]
        landmarks_final = [landmarks[i][:68] for i in range(len(landmarks)) if index[i]]
        batch = batch_copy[index]

        torch.cuda.empty_cache()

        return landmarks_final, batch, indices_final

    def _align_batch(self, batch, landmarks):
        self.logger.debug("Transforming batch")
        retval = []

        # align batches to images with dimension of model_output_size
        size = len(landmarks)
        matrices = self._get_matrices(landmarks)

        for frame in range(size):
            img = batch[frame]
            mats = matrices[frame]
            warps = []
            # need to loop here as one frame can contain multiple faces, thus multiple affine matrices
            for matrix in mats:
                warp = cv2.warpAffine(img, matrix, (512, 512), cv2.INTER_AREA)
                warps.append(warp)
            retval.append(warps)
        
        return retval, matrices

    def _get_matrices(self, landmarks):

        # compute alignment matrices from landmarks based on chosen centering
        self.logger.debug("Get alignment matrices")

        retval = []

        for landmark in landmarks:
            matrices = []
            num_faces = landmark.shape[0] // 68
            # need to loop here as frame can contain multiple faces and thus multiple detected landmarks
            for face in range(num_faces):
                lm = landmark[(face*68):((face+1)*68)]

                legacy = umeyama(lm[17:], _MEAN_FACE, True)[0:2]

                lms = cv2.transform(np.expand_dims(lm, axis=1), legacy).squeeze()
                pose = PoseEstimate(lms)
                matrix = legacy.copy() 
                matrix[:, 2] -= pose.offset["head"]
                matrix = matrix*(512-2*160)
                matrix[:, 2] += 160

                # # correct matrix such that warped image will have the desired dimensions
                # correction = np.zeros_like(matrix)
                # correction[:,0] = [1/1*self.output_size/512, 0]
                # correction[:,1] = [0,1/1*self.output_size/512]
                # correction[:,2] -= (1-1)/2*self.output_size

                # matrix[:,:2] = np.matmul(correction[:,:2], matrix[:,:2])
                # matrix[:, 2] = np.matmul(correction[:, :2], matrix[:, 2]) + correction[:, 2]

                matrices.append(matrix)

            retval.append(matrices)    

        return retval   
    
    # masking
    def _prep_batch_for_masks(self, batch, num_faces):
        # prep batch to compute 512px mask for each image. The input image of the network has to be a 512px aligned image, thus the alignment in the loop.
        self.logger.debug("Prep batch for masking")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        tensor = torch.zeros(num_faces, 3, 512, 512)
        counter = 0
        for frame in batch:
            for img in frame:
                tensor[counter] = transform(img)
                counter += 1

        return tensor

    def _clean_masks(self, masks):
        # we clean the masks here as we dont need the full segmentation in the conversion phase
        self.logger.debug("Cleaning batch masks...")
        face_points = np.logical_or(np.logical_and(masks<7, masks>0), np.logical_and(masks>9, masks<14)).astype(int)
        mouth_points = np.logical_or(masks==11, np.logical_or(masks==12, masks==13)).astype(int)*2
        eye_points = np.logical_or(masks==4, masks==5).astype(int)
        brow_points = (masks==2).astype(int)*3

        clean_mask = face_points + mouth_points + eye_points + brow_points

        return clean_mask.astype(np.uint8)

    def _get_masks(self, batch):
        # compute face masks from batch
        self.logger.debug("Predict face masks")

        num_faces = 0
        for frame in batch:
            num_faces += len(frame)

        batch_tensor = self._prep_batch_for_masks(batch, num_faces)
        batch_tensor = batch_tensor.to(self.device)
        
        # this whole block until cleanmasks computes the masks and handles the case when there are more faces than batch_size 
        indices = [0]
        index = 0
        masks = np.zeros((batch_tensor.shape[0], batch_tensor.shape[2], batch_tensor.shape[3]))

        while index < batch_tensor.shape[0]:
            index += self.batch_size
            indices.append(min(index, batch_tensor.shape[0]))

        for i in range(len(indices)-1):
            input = batch_tensor[indices[i]:indices[i+1]]
            masks[indices[i]:indices[i+1]] = self.mask_net(input)[0].cpu().numpy().argmax(1)

        masks = self._clean_masks(masks)

        retval_masks = []
        counter = 0

        for index in range(len(batch)):
            n = len(batch[index])
            m = []
            for _ in range(n):
                m.append(masks[counter])
                counter += 1

            retval_masks.append(m)

        torch.cuda.empty_cache()

        return retval_masks