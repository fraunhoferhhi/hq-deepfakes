import os
import logging

import cv2
import numpy as np

import torch
import face_alignment

from ..conversion.utils.convutils import load_video, umeyama, setup_logger, PoseEstimate, _MEAN_FACE


VIDEO_TYPES = [".mp4", ".avi"]
IMAGE_TYPES = [".png", ".jpg", ".jpeg"]


# standard face extractor
class FaceExtractor():
    '''
    FaceExtractor:
    Extracts faces from all video or image files in a given path, aligns and saves them. Path can also point to a singular video or png file (use respective methods then).
    args:
        centering: How to align the images, currently only "head" is supported.
        device: Which device to use for deep learning models, cuda or cpu.
        output_size: Output dimension of the extracted face images in pixels, 512 is recommended.
        batch_size: Batch size to use for deep learning models.
        every_nth_frame: In case of videos, whether to extract the entire video or only every nth frame (usefull for very large videos in order to save memory)
        only_top_face: Whether to extract only the face with the largest bounding box / certainty. (Do not set to true if you want to handle videos with multiple relevant faces)
        verbose: Whether to log information of the extration process and its setup
    '''
    def __init__(self, 
                 centering:str = "head", 
                 device:str = "cuda",
                 output_size:int = 512,
                 batch_size:int = 8,
                 every_nth_frame:int = 1,
                 num_frames:int = -1,
                 only_top_face:bool = False,
                 verbose:bool = True):
 
        self.logger = setup_logger(verbose)

        # save args as attributes
        self.centering = centering
        self.device = device
        self.output_size = output_size
        self.batch_size = batch_size
        self.every_nth_frame = every_nth_frame
        self.num_frames = num_frames
        self.only_top_face = only_top_face
        self.verbose = verbose

        # load face alignment network
        self.logger.info("Loading face alignment network...")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)

        self.logger.info("Done!")


    # call this function to run the extractor on the "entire" data path
    def process(self,
                data_path: str,
                out_path: str = None):
        '''
        Main function of the extractor, extracts faces from all video and image files in data_path and saves them in out_path.
        args:
            data_path: directory that holds the media to extract the faces from.
            out_path: (Optional) where to save the extracted faces, if not set a "faces" directory will be created inside data_path
        
        '''

        assert os.path.isdir(data_path), "data_path must be a directory, if you want to pass paths to single image or video files use process_image or process_video methods!"

        # handle paths / preprocess
        out_path = self._preprocess(data_path, out_path)

        self.logger.info("Start processing")

        input_videos = [os.path.join(data_path, f) for f in os.listdir(data_path) if f[-4:] in VIDEO_TYPES and f not in self.seen]
        input_images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f[-4:] in IMAGE_TYPES and f not in self.seen]

        self.logger.info("Processing {} videos and {} images, {} files already processed previously".format(len(input_videos), len(input_images), len(self.seen)))

        for path in input_videos:
            self.process_video(path, out_path, log_to_seen=True)
        for path in input_images:
            self.process_image(path, out_path, log_to_seen=True)

        self.logger.info("Finished processing")

    def process_image(self, 
                      img_path: str, 
                      out_path: str = None,
                      log_to_seen: bool = False):
        '''
        Function to extract faces from an image file.
        args:
            img_path: path to the image file to extract faces from
            out_path: (Optional) where to store the extracted faces, if not specified faces will be saved next to the image
            log_to_seen: whether to log if the image has already been processed (useful when you extract from large datasets and need to restart the process)
        '''

        assert img_path[-4:] in IMAGE_TYPES, "File type not supported, ensure that you passed the path to a png or jpg"

        img = cv2.imread(img_path)
        indices = [0]

        out_path = out_path if out_path is not None else img_path.rpartition("/")[0]

        self.logger.info("Processing image: {}".format(img_path))

        batch = np.expand_dims(img, 0)
        landmarks, batch, indices = self._detect_from_batch(batch, indices)
        if len(landmarks) == 0:
            print("No face detected!")
        else:
            # align images
            batch = self._align_batch(batch, landmarks)
            # save images
            self._save_batch(batch, indices, img_path, out_path)

        if log_to_seen:
            with open(self.seen_path, "a") as f:
                f.write(img_path.rpartition("/")[-1])
                f.write('\n')

    def process_video(self, 
                      video_path: str,
                      out_path: str = None,
                      log_to_seen: bool = False):
        
        '''
        Function to extract faces from an image file.
        args:
            video_path: path to the video file to extract faces from
            out_path: (Optional) where to store the extracted faces, if not specified faces will be saved next to the video
            log_to_seen: whether to log if the video has already been processed (useful when you extract from large datasets and need to restart the process)
        '''

        assert video_path[-4:] in VIDEO_TYPES, "File type not supported, ensure that you passed the path to an mp4 or avi"


        self.logger.info("Processing video: {}".format(video_path))
        video = load_video(video_path, every_nth_frame=self.every_nth_frame, verbose=self.verbose, num_frames=self.num_frames)

        out_path = out_path if out_path is not None else video_path.rpartition("/")[0]
     
        num_frames = video.shape[0]
        for i in range(0, num_frames, self.batch_size):
            start = i
            end = i+self.batch_size if i+self.batch_size < num_frames else num_frames
            indices = list(range(start, end))

            batch = video[start:end]
            landmarks, batch, indices = self._detect_from_batch(batch, indices)
            indices = [f*self.every_nth_frame for f in indices]

            # handle case if no face was detected
            if len(landmarks) == 0:
                continue

            # align images
            batch = self._align_batch(batch, landmarks)

            # save images
            self._save_batch(batch, indices, video_path, out_path)

        if log_to_seen:
            with open(self.seen_path, "a") as f:
                f.write(video_path.rpartition("/")[-1])
                f.write('\n')

    ## helper stuff
    def _preprocess(self, data_path: str, out_path: str = None):
        # handle paths / preprocess
        self.seen_path = os.path.join(data_path, "seen_face.txt")
        if not os.path.exists(self.seen_path):
            f = open(self.seen_path, "x")
            seen = []
        else:
            self.logger.info("Loading seen file from: {}".format(self.seen_path))
            f = open(self.seen_path, "r")
            seen = f.read()
            seen = seen.split('\n')
        self.seen = seen

        if out_path is None:
            dp = data_path
            out_path = os.path.join(dp, "faces") 

        if not os.path.exists(out_path):
            self.logger.info("Creating out path at: {}".format(out_path))
            os.makedirs(out_path)

        return out_path
    
    def _save_batch(self, batch, indices, path, save_dir):
        logging.debug("Saving batch")
        size = len(batch)
        video_name = path.split("/")[-1][:-4]

        for frame in range(size):
            index = indices[frame]
            images = batch[frame]
            counter = 0
            # multiple faces can be in one frame, thus we loop here and indicate the frame and face index in the suffix
            for img in images:
                suffix = "_{}_{}.png".format(str(index).zfill(5), counter)
                save_path = os.path.join(save_dir, video_name + suffix)
                cv2.imwrite(save_path, img)
                counter += 1

    # landmarks
    def _detect_from_batch(self, batch, indices):
        logging.debug("Detecting landmarks")

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
                index.append(True)

        # get valid landmarks, images and indices (latter are only for saving with the right frame index)
        indices_final = [indices[i] for i in range(len(indices)) if index[i]]

        if self.only_top_face:
            landmarks_final = [landmarks[i][:68] for i in range(len(landmarks)) if index[i]]
            # boxes_final = [[boxes[i][0]] for i in range(len(boxes)) if index[i]]
        else:
            landmarks_final = [landmarks[i] for i in range(len(landmarks)) if index[i]]
            # boxes_final = [boxes[i] for i in range(len(boxes)) if index[i]]

        batch = batch_copy[index]

        torch.cuda.empty_cache()

        return landmarks_final, batch, indices_final

    def _prep_batch_for_detect(self, batch):
        # transform batch to tensor in order to compute landmarks
        logging.debug("Prep batch for detection")

        size, h, w = batch.shape[:3]
        tensor = torch.zeros(size, 3, h, w)

        for i in range(size):
            img = np.moveaxis(batch[i], -1 , 0)
            tensor[i] = torch.from_numpy(img)

        return tensor

    # alignment
    def _align_batch(self, batch, landmarks):
        logging.debug("Transforming batch")
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
                warp = cv2.warpAffine(img, matrix, (self.output_size, self.output_size), cv2.INTER_AREA)
                warps.append(warp)
            retval.append(warps)
        
        return retval

    def _get_matrices(self, landmarks):
        # compute alignment matrices from landmarks based on chosen centering
        logging.debug("Get alignment matrices")

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
                matrix[:, 2] -= pose.offset[self.centering]
                matrix = matrix*(512-2*160)
                matrix[:, 2] += 160

                # correct matrix such that warped image will have the desired dimensions
                correction = np.zeros_like(matrix)
                correction[:,0] = [1/1*self.output_size/512, 0]
                correction[:,1] = [0,1/1*self.output_size/512]
                correction[:,2] -= (1-1)/2*self.output_size

                matrix[:,:2] = np.matmul(correction[:,:2], matrix[:,:2])
                matrix[:, 2] = np.matmul(correction[:, :2], matrix[:, 2]) + correction[:, 2]

                matrices.append(matrix)

            retval.append(matrices)    

        return retval   
