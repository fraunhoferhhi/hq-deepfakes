import os
import shutil
import yaml
import zlib
from pathlib import Path
from typing import Union

import torch
from torchvision import transforms
import cv2
import numpy as np

from ..training.model import custom
from .videoprepper import VideoPrepper
from .utils.convutils import setup_logger, load_alignments, load_video, save_video

from tqdm import tqdm

class Converter():
    '''
    Conversion class for deepfake models. Loads a specific deepfake model and then works for that model only. Can process singular videos (swaps the largest face in there).
    args:
        model_ckpt: path that holds the model checkpoint.
        model_config: path to the config that was used to build the autoencoder
        batch_size: batch_size of autoencoder forward passes
        pad: padding that is used prior to mask squeezing (this reduces blending artifacts). Can be a list of 4 (top, bot, left right) or single value for all sides
        adjust_color: whether to adjust the colour average of the swap wrt to the input (not necessary since poisson blending also does this)
        writer: whether to write the fake as mp4 or pngs (additional option: mp4-crf14 saves the videos with a lower compression factor than mp4)
        device: device to run autoencoder on
        verbose: whether to log some minor progress statements
    '''
    def __init__(self,
                 model_ckpt: str,
                 model_config: str,
                 batch_size: int = 8,
                 pad: list = [30],
                 adjust_color: bool = False,
                 writer: str = "mp4",
                 device: str = "cuda",
                 verbose: bool = False):
        
        self.logger = setup_logger(verbose)

        self.model_ckpt = model_ckpt
        self.model_config = model_config

        self.pad = pad
        self.adjust_color = adjust_color
        self.writer = writer

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        self._load_model()

        # we need the video prepper here for 2 reasons: 1. to prep videos if they have not been prepped yet 2. to get the face segmentation masks of the fakes (for better blending)
        self.vp = VideoPrepper(self.device)

        self.tt = transforms.ToTensor()

    ## setup
    def _load_model(self):
        self.logger.info("Loading model config from {}".format(self.model_config))
        model_config = yaml.safe_load(Path(self.model_config).read_text())

        # scrap some info from the model config
        self.model_img_size = int(model_config["data"]["model_img_size"])
        self.coverage_ratio = float(model_config["data"]["coverage_ratio"])

        self.logger.info("Loading model state from {}".format(self.model_ckpt))
        state_dict = torch.load(self.model_ckpt)["state_dict"]
        new_state_dict = {}

        # clean statedict
        for k,v in state_dict.items():
            new_key = k[3:]
            new_state_dict[new_key] = v

        model = custom(model_config["model"]["cfg"])
        model.load_state_dict(new_state_dict)
        model.eval()
        model.to(self.device)

        self.model = model

    ## processing
    def process_video(self,
                      video_path: str,
                      direction: str,
                      prep_path: str = None,
                      out_dir: str = None):
        '''
        video_path: path to video to convert
        direction: whether to swap from A to B or B to A
        prep_path (optional): path to _prep.fsa file. If not specified, will look for the file next to the video. If not there, one will be generated.
        '''
        # manage alignments
        prep_path = prep_path if prep_path is not None else video_path.split(".")[0] + "_prep.fsa"
        if not os.path.exists(prep_path):
            self.vp.process_video(video_path)

        alignments = load_alignments(prep_path)

        # load video, build fake video
        video_name = video_path.split(".")[0].split("/")[-1]
        video, fourcc, fps = load_video(video_path, ret_info=True)
        fake_video = np.zeros_like(video)

        num_frames = video.shape[0]

        self.logger.info("Start processing of {} frames".format(num_frames))
        # convert in batches
        with torch.no_grad():
            for i in tqdm(range(0, num_frames, self.batch_size)):
                start = i
                end = min(i + self.batch_size, num_frames)

                # pre blending performs forward pass through model and obtains new masks based on the intersection of old and newly computed masks
                new_faces, old_faces, masks = self._pre_blending(video, video_name, alignments, start, end, direction)

                for j in range(start, end):
                    frame = video[j]
                    new_face = new_faces[j - start]
                    old_face = old_faces[j - start]
                    mask = masks[j - start]

                    placeholder = frame.copy()
                    background = frame.copy()

                    key = "{}_{}.png".format(video_name, j)
                    if key not in alignments.keys():
                        fake_video[j] = frame
                    else:
                        matrix = alignments[key]["matrix"]
                        # pre warp adjustments / plugins
                        new_face, mask = self._pre_warp_adjustment(old_face, new_face, mask)
                        # align fake and mask onto base img
                        patch, mask_aligned = self._patch_images(placeholder, new_face, matrix, mask)
                        # blend fake patch with background
                        blended = self._blend_images(patch, background, mask_aligned)
                        # save blended 
                        fake_video[j] = blended

        self._save_video(fake_video, video_path, None, fps, out_dir)

    def _pre_blending(self, video, video_name, alignments, start, end, direction):
        old_faces = np.zeros((end - start, 256, 256, 3), dtype=np.float32)

        # obtain (aligned) old faces and new faces through forward pass  
        input_tensor = torch.zeros(end - start, 3, 256, 256)
        for frame_index in range(start, end):
            key = "{}_{}.png".format(video_name, frame_index)

            if not key in alignments.keys():
                continue

            matrix = alignments[key]["matrix"]
            frame = video[frame_index]

            # get input for model
            adjusted_matrix = self._adjust_matrix(matrix, target_size=self.model_img_size)
            input  = cv2.warpAffine(frame, adjusted_matrix, (self.model_img_size, self.model_img_size), flags=cv2.INTER_AREA)
            input_tensor[frame_index - start] = self.tt(input)

            # reset old face for latter color adjustments (such that the dims are equal to model output size)
            adjusted_matrix = self._adjust_matrix(matrix, target_size=self.model_img_size)
            old_face  = cv2.warpAffine(frame, adjusted_matrix, (self.model_img_size, self.model_img_size), flags=cv2.INTER_AREA)
            old_faces[frame_index - start] = old_face

        # build input tensor for autoencoder
        input_tensor = input_tensor.to(self.device)
        output = self.model.forward_from_to(input_tensor, dec = direction)
        output = output.cpu().numpy()
        output = np.moveaxis(output, 1, -1)

        new_faces = output*255

        # get masks
        masks = self._adjust_masks(new_faces, video_name, alignments, start, end)

        return new_faces, old_faces, masks


    ## util
    # matrix
    def _adjust_matrix(self, matrix, target_size):
        # adjust the alignment matrix to respect coverage and target size
        mat = matrix.copy()

        correction = np.zeros_like(mat)
        correction[:,0] = [1/self.coverage_ratio*target_size/512, 0]
        correction[:,1] = [0,1/self.coverage_ratio*target_size/512]
        correction[:,2] -= (1-self.coverage_ratio)/2*target_size

        mat[:,:2] = np.matmul(correction[:,:2], mat[:,:2])
        mat[:, 2] = np.matmul(correction[:, :2], mat[:, 2]) + correction[:, 2]

        return mat
    
    # masking
    def _adjust_masks(self, new_faces, video_name, alignments, start, end):
        old_masks = self._get_old_masks(video_name, alignments, start, end) # sample old masks from prep dict
        new_masks = self._get_new_masks(new_faces) # compute new masks based on forward pass of swaps through masking model in videoprepper

        intersection = np.minimum(new_masks, old_masks)

        if len(self.pad) == 1:
            pad = self.pad[0]
            pad_top = int(pad) // 2
            pad_bot = int(pad) // 2
            pad_left = int(pad) // 2
            pad_right = int(pad) // 2
        else:
            pad_top = int(self.pad[0])
            pad_bot = int(self.pad[1])
            pad_left = int(self.pad[2])
            pad_right = int(self.pad[3])

        
        # pad and squeeze the masks
        squeezed = np.zeros((new_faces.shape[0], 512 + pad_top + pad_bot, 512 + pad_left + pad_right), dtype=np.uint8)
        start_y = pad_top
        end_y = 512 + pad_top
        start_x = pad_left
        end_x = 512 + pad_left

        squeezed[:, start_y:end_y, start_x:end_x] = intersection

        # masks will be resized to proper size later!
        masks = np.zeros((new_faces.shape[0], 512, 512), dtype=np.uint8)
        for index in range(new_faces.shape[0]):
            masks[index] = cv2.resize(squeezed[index], (512, 512), cv2.INTER_AREA)

        return masks

    def _get_old_masks(self, video_name, alignments, start, end):
        coverage = int(512 * self.coverage_ratio // 2) * 2
        tgt_slices = slice(512 // 2 - coverage // 2,
                            512 // 2 + coverage // 2)


        # load and resize masks (respects coverage ratio)
        old_masks = np.zeros((end - start, 512, 512), np.uint8)
        for frame_index in range(start, end):
            key = "{}_{}.png".format(video_name, frame_index)
            if key in alignments.keys():
                mask = alignments[key]["mask"]
                mask = np.frombuffer(zlib.decompress(mask), dtype="uint8").reshape((512, 512))
                mask = cv2.resize(mask[tgt_slices, tgt_slices], (512, 512), interpolation=cv2.INTER_CUBIC)
                old_masks[frame_index - start] = mask

                old_masks = np.clip(old_masks, 0, 1)
        
        return old_masks
    
    def _get_new_masks(self, new_faces):
        # prep transform for masking network
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # get new masks
        new_masks = np.zeros((new_faces.shape[0], 512, 512), dtype=np.uint8)
        input_tensor = torch.zeros((new_faces.shape[0], 3, 512, 512))

        for index in range(new_faces.shape[0]):
            face = new_faces[index]
            face = transform(face)
            input_tensor[index] = face

        input_tensor = input_tensor.to(self.device)
        new = self.vp.mask_net(input_tensor / 255.)
        new = new[0].cpu().numpy().argmax(1)
        new = np.logical_or(np.logical_and(new<7, new>0), np.logical_and(new>9, new<14)).astype(np.uint8)
        new_masks = new

        return new_masks
    
    # blending
    def _pre_warp_adjustment(self, old_face, new_face, mask):
        # apply mask blurring and color averaging to mask and fake face
        mask = self._prep_mask(mask)
        new_face = self._adjust_color(old_face, new_face, mask) if self.adjust_color else new_face

        return new_face, mask

    def _prep_mask(self, mask, threshold=4, kernel_size=3, passes=4, blur_type="normalized", resize=True):
        # apply multiple passes of normalized blurring to the mask
        factor = 0.5 if blur_type == "normalized" else 0.8
        kernel = (kernel_size, kernel_size)
        blurred = mask.copy().astype(np.float32)
        for _ in range(passes):
            blurred = cv2.blur(blurred, kernel)
            kernel_size = int(round(kernel[0] * factor))
            kernel_size += 1 if kernel_size % 2 == 0 else 0
            kernel = (kernel_size, kernel_size)

        # resize to model img size
        if resize:
            blurred = cv2.resize(blurred, (self.model_img_size, self.model_img_size), interpolation=cv2.INTER_AREA)

        return blurred

    def _adjust_color(self, old_face, new_face, mask):
        # equalize the color averages of the input face and swapped face in the masked region
        for _ in [0,1]:
            diff = old_face - new_face
            avg_diff = np.sum(diff*np.expand_dims(mask, axis=-1), axis=(0,1))
            adjustment = avg_diff / np.sum(mask, axis=(0,1))
            new_face += adjustment

        new_face = np.clip(new_face, 0, 255)
        return new_face
    
    def _patch_images(self, img, fake, matrix, mask):
        # aligning the fake image and the mask to the original input frame
        fake = fake.astype(np.uint8)
        frame_size = (img.shape[1], img.shape[0])
        mat = self._adjust_matrix(matrix, self.model_img_size)
        patch = cv2.warpAffine(src=fake,
                                M=mat,
                                dsize=frame_size,
                                dst=img,
                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_TRANSPARENT)

        mask = mask*255
        mask = np.dstack((mask, mask, mask))
        ph = np.zeros_like(img).astype(np.float32)
        mask_aligned  = cv2.warpAffine(src=mask,
                                        M=mat,
                                        dsize=frame_size,
                                        dst=ph,
                                        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_TRANSPARENT)

        return patch, mask_aligned[:, :, 0]/255

    def _blend_images(self, patch, background, mask):
        # blend background and foreground (patch) images with mask
        mask = (mask*255).astype(np.uint8)

        flag = cv2.NORMAL_CLONE
        br = cv2.boundingRect(mask)
        center = (br[0] + br[2] // 2, br[1] + br[3] // 2)

        try:
            blended = cv2.seamlessClone(patch, background, mask, center, flag)
        except:
            return background.astype(np.float32)

        return blended

    # saving
    def _save_video(self, fake_video, video_path, fourcc, fps, out_dir=None):
        model_name = self.model_ckpt.split("/")[-1].split(".")[0]
        model_name += "_padding"
        for p in self.pad:
            model_name += "-{}".format(p)
            
        parent_dir = video_path.rpartition("/")[0]
        video_name = video_path.rpartition("/")[-1]

        if out_dir is None:
            out_dir = os.path.join(parent_dir, "fakes", model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, video_name)

        if self.writer == "mp4":
            save_video(fake_video, out_path, "mp4", fourcc, fps)

        else:
            save_video(fake_video, out_path, "png", fourcc, fps)


            if self.writer == "mp4-crf14":
                frame_path = out_path.replace(".mp4", "_frames")
                cmd = "ffmpeg -framerate {} -pattern_type glob -i '{}/*.png' -c:v libx264 -preset slow -crf 14 -pix_fmt yuv420p {}".format(fps, frame_path, out_path)

                self.logger.info("Finalizing with: {}".format(cmd))
                os.system(cmd)

                self.logger.info("Removing {}".format(frame_path))
                shutil.rmtree(frame_path)



