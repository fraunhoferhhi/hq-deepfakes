import cv2

import numpy as np


import torch
from torch.nn import MSELoss, L1Loss

from piq import ssim

#### MAIN
class DeepfakeLoss():
    def __init__(self, 
                 eye_multiplier: int = 3, 
                 mouth_multiplier: int = 2, 
                 blur: bool = True, 
                 l2_weight: float = 1, 
                 l1_weight: float = 0, 
                 ffl_weight: float = 0, 
                 ffl_alpha:float = 1.0, 
                 ffl_mask: bool = True, 
                 lpips_weight: float = 0, 
                 lpips_net: str = "vgg", 
                 lpips_mask: bool = True):
        
        self.ffl_weight = ffl_weight
        self.ffl_mask = ffl_mask
        self.lpips_weight = lpips_weight
        self.lpips_mask = lpips_mask

        self.loss_recon = ReconstructionLoss(l2_weight=l2_weight, l1_weight=l1_weight, eye_multiplier=eye_multiplier, mouth_multiplier=mouth_multiplier, blur=blur)
        
        if ffl_weight > 0:
            self.loss_ffl = FocalFrequencyLoss(alpha=ffl_alpha)
        
        # if lpips_weight > 0:
        #     self.loss_lpips = LPIPS(net=lpips_net, model_path=WEIGHTS_PATH)

    def compute(self, X, y, m):
        loss_dict = {}
        loss_dict["ReconstructionLoss"] = self.loss_recon.compute(X, y, m)

        if self.ffl_weight > 0:
            if self.ffl_mask:
                X_in = mask_input(X, m, mode="face", blur=False)
                y_in = mask_input(y, m, mode="face", blur=False)
            else:
                X_in, y_in = X, y

            loss_dict["FocalFrequencyLoss"] = self.loss_ffl(y_in, X_in)

        # if self.lpips_weight > 0:
        #     if self.lpips_mask:
        #         X_in = mask_input(X, m, mode="face", blur=False)
        #         y_in = mask_input(y, m, mode="face", blur=False)
        #     else:
        #         X_in, y_in = X, y     

        #     X_in = (X_in - 0-5)*2
        #     y_in = (y_in - 0.5)*2
        #     # self.loss_lpips.to(X_in.device)
        #     loss_dict["LPIPS"] = torch.mean(self.loss_lpips.forward(X_in, y_in))

        return loss_dict

#### RECONSTRUCTION 
class ReconstructionLoss():
    '''
    Reconstruction Loss, used to compute the loss for both sides a and b at once
    inputs:
        loss_fn: standard loss function for computations (default: dssim)
        regularization_weight: weight for l2 regularization of every loss_fn call, dont use when loss_fn is not ssim or gmsd (default 100.0)
        eye_multiplier: weight for eye-specific loss (default: 3)
        mouth_multiplier: weight for mouth-specific loss (default: 2)
        blur: If True, enables mask bluring to avoid sharp edges, currently not working (default: False)

    funcs:
        compute: computes the masked loss_fn for a batch
        get: computes the masked loss_fn for both sides of the model 
    '''
    def __init__(self, 
                 l2_weight:float=1.0, 
                 l1_weight:float=0, 
                 eye_multiplier:int=3, 
                 mouth_multiplier:int=2, 
                 blur:bool=True):
        

        self.loss_fn = dssim 
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.eye_multiplier = eye_multiplier
        self.mouth_multiplier = mouth_multiplier
        self.blur = blur

    def compute(self, X, y, m):
        loss = mask_loss(X, y, m, self.loss_fn, "face", self.l2_weight, self.l1_weight, self.blur)
        if self.eye_multiplier > 0:
            loss += mask_loss(X, y, m, self.loss_fn, "eye", self.l2_weight, self.l1_weight, self.blur)*self.eye_multiplier
        if self.mouth_multiplier > 0:
            loss += mask_loss(X, y, m, self.loss_fn, "mouth", self.l2_weight, self.l1_weight, self.blur)*self.mouth_multiplier

        return loss 

    def get(self, pred_a, pred_b, y_a, y_b, m_a, m_b):
        # computes reconstruction loss based on predictions, targets and masks
        loss_a = self.compute(pred_a, y_a, m_a)
        loss_b = self.compute(pred_b, y_b, m_b)

        return loss_a, loss_b
            


#### FREQUENCY
# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(torch.nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight





#### UTILITY
### mathematical loss functions: dssim, mse 
def dssim(X, y):
    loss = (1-ssim(X,y, data_range=1., kernel_size=3))/2.0
    return loss


### mask stuff:
def mask_loss(X, y, m, loss_fn, mode="face", l2_weight=0., l1_weight=0.25, blur=True):

    X_masked = mask_input(X, m, mode, blur)
    targets_masked = mask_input(y, m, mode, blur)

    loss = loss_fn(X_masked, targets_masked)

    if l1_weight > 0:
        mae = L1Loss()
        loss += mae(X_masked, targets_masked)*l1_weight

    if l2_weight > 0:
        mse = MSELoss()
        loss += mse(X_masked, targets_masked)*l2_weight

    return loss

def mask_input(x, m, mode ="face", blur=True):
    if mode == "face":
        mask = (m > 0)*1.0
    elif mode == "eye":
        mask = (m == 10)*1.0
    elif mode == "mouth":
        mask = (m == 20)*1.0

    if blur:
        mask = gaussian_blur(mask)

    mask = mask.unsqueeze(1)
    x_masked = x*mask

    return x_masked

def gaussian_blur(m, kernel_size=3, sigma=1):

    mask = m.detach().cpu().numpy()
    mask = np.moveaxis(mask, 0, -1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), sigma)
    mask = np.moveaxis(mask, -1, 0)
    mask = torch.from_numpy(mask).to("cuda")

    return mask

