import os
import logging
import pickle 
import zlib 
import json

import cv2
import numpy as np
import torch

def setup_logger(verbose):
    logger = logging.getLogger(__name__)
    logger.setLevel(20 if verbose else 30)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def load_alignments(path, verbose=False):
    if verbose:
        print("Loading alignments from {}".format(path))

    if path.endswith(".fsa"):
        with open(path, "rb") as s_file:
            data = s_file.read()
        alignments = pickle.loads(zlib.decompress(data))

    elif path.endswith(".json"):
        with open(path, "rb") as s_file:
            alignments = json.load(s_file)

    return alignments

def save_dict(alignments, path, verbose=False, file_type="fsa"):
    if verbose:
        print("Saving information file to: {}".format(path))

    if file_type == "json":
        path += ".json"
        with open(path, "w") as s_file:
            json.dump(alignments, s_file, indent=1)
    else:
        path += ".fsa"
        alignment = zlib.compress(pickle.dumps(alignments))
        with open(path, "wb") as s_file:
            s_file.write(alignment)

def load_video(path_to_video, every_nth_frame=1, verbose=False, ret_info=False, num_frames=-1):
    if verbose:
        print("Loading video from: {}".format(path_to_video))

    cap = cv2.VideoCapture(path_to_video)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames != -1:
        frameCount = min(num_frames, frameCount)

    buf = []

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, frame = cap.read()

        if fc % every_nth_frame == 0:
            if frame is not None:
                buf.append(frame)
        fc += 1

    cap.release()

    buf = np.stack(buf).astype(np.uint8)

    if verbose:
        print("Video loaded!")

    if ret_info:
        return buf, fourcc, fps
    else:
        return buf
    
def save_video(video, path, writer, fourcc=None, fps=None):
    if writer == "mp4":
        print("Save fake video to {}".format(path))
        h, w = video[0].shape[:2]

        fourcc = fourcc if fourcc is not None else cv2.VideoWriter_fourcc("m", "p", "4", "v")
        fps = fps if fps is not None else 24

        out = cv2.VideoWriter(path, fourcc, fps, (w,h))

        for frame in range(video.shape[0]):
            out.write(video[frame])
        
        out.release()

    elif writer == "png":
        outfile = path.split(".")[0] + "_frames"
        if not os.path.exists(outfile):
            os.makedirs(outfile, exist_ok=True)

        print("Save fake video frames to {}".format(outfile))
        for frame in range(video.shape[0]):
            name = str(frame).zfill(6) + ".png"
            p = os.path.join(outfile, name)
            cv2.imwrite(p, video[frame])


class PoseEstimate():
    """ Estimates pose from a generic 3D head model for the given 2D face landmarks.

    Parameters
    ----------
    landmarks: :class:`numpy.ndarry`
        The original 68 point landmarks aligned to 0.0 - 1.0 range

    References
    ----------
    Head Pose Estimation using OpenCV and Dlib - https://www.learnopencv.com/tag/solvepnp/
    3D Model points - http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    """
    def __init__(self, landmarks):
        self._distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
        self._xyz_2d = None

        self._camera_matrix = self._get_camera_matrix()
        self._rotation, self._translation = self._solve_pnp(landmarks)
        self._offset = self._get_offset()
        self._pitch_yaw = None

    @property
    def xyz_2d(self):
        """ :class:`numpy.ndarray` projected (x, y) coordinates for each x, y, z point at a
        constant distance from adjusted center of the skull (0.5, 0.5) in the 2D space. """
        if self._xyz_2d is None:
            xyz = cv2.projectPoints(np.float32([[6, 0, -2.3], [0, 6, -2.3], [0, 0, 3.7]]),
                                    self._rotation,
                                    self._translation,
                                    self._camera_matrix,
                                    self._distortion_coefficients)[0].squeeze()
            self._xyz_2d = xyz - self._offset["head"]
        return self._xyz_2d

    @property
    def offset(self):
        """ dict: The amount to offset a standard 0.0 - 1.0 umeyama transformation matrix for a
        from the center of the face (between the eyes) or center of the head (middle of skull)
        rather than the nose area. """
        return self._offset

    @property
    def pitch(self):
        """ float: The pitch of the aligned face in eular angles """
        if not self._pitch_yaw:
            self._get_pitch_yaw()
        return self._pitch_yaw[0]

    @property
    def yaw(self):
        """ float: The yaw of the aligned face in eular angles """
        if not self._pitch_yaw:
            self._get_pitch_yaw()
        return self._pitch_yaw[1]

    def _get_pitch_yaw(self):
        """ Obtain the yaw and pitch from the :attr:`_rotation` in eular angles. """
        proj_matrix = np.zeros((3, 4), dtype="float32")
        proj_matrix[:3, :3] = cv2.Rodrigues(self._rotation)[0]
        euler = cv2.decomposeProjectionMatrix(proj_matrix)[-1]
        self._pitch_yaw = (euler[0][0], euler[1][0])
        # print("yaw_pitch: %s", self._pitch_yaw)

    @classmethod
    def _get_camera_matrix(cls):
        """ Obtain an estimate of the camera matrix based off the original frame dimensions.

        Returns
        -------
        :class:`numpy.ndarray`
            An estimated camera matrix
        """
        focal_length = 4
        camera_matrix = np.array([[focal_length, 0, 0.5],
                                  [0, focal_length, 0.5],
                                  [0, 0, 1]], dtype="double")
        # print("camera_matrix: %s", camera_matrix)
        return camera_matrix

    def _solve_pnp(self, landmarks):
        """ Solve the Perspective-n-Point for the given landmarks.

        Takes 2D landmarks in world space and estimates the rotation and translation vectors
        in 3D space.

        Parameters
        ----------
        landmarks: :class:`numpy.ndarry`
            The original 68 point landmark co-ordinates relating to the original frame

        Returns
        -------
        rotation: :class:`numpy.ndarray`
            The solved rotation vector
        translation: :class:`numpy.ndarray`
            The solved translation vector
        """
        points = landmarks[[6, 7, 8, 9, 10, 17, 21, 22, 26, 31, 32, 33, 34,
                            35, 36, 39, 42, 45, 48, 50, 51, 52, 54, 56, 57, 58]]
        _, rotation, translation = cv2.solvePnP(_MEAN_FACE_3D,
                                                points,
                                                self._camera_matrix,
                                                self._distortion_coefficients,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        # print("points: %s, rotation: %s, translation: %s", points, rotation, translation)
        return rotation, translation

    def _get_offset(self):
        """ Obtain the offset between the original center of the extracted face to the new center
        of the head in 2D space.

        Returns
        -------
        :class:`numpy.ndarray`
            The x, y offset of the new center from the old center.
        """
        offset = dict(legacy=np.array([0.0, 0.0]))
        points = dict(head=(0, 0, -2.3), face=(0, -1.5, 4.2))

        for key, pnts in points.items():
            center = cv2.projectPoints(np.float32([pnts]),
                                       self._rotation,
                                       self._translation,
                                       self._camera_matrix,
                                       self._distortion_coefficients)[0].squeeze()
            # print("center %s: %s", key, center)
            offset[key] = center - (0.5, 0.5)
        # print("offset: %s", offset)
        return offset
    

_MEAN_FACE = np.array([[0.010086, 0.106454], [0.085135, 0.038915], [0.191003, 0.018748],
                       [0.300643, 0.034489], [0.403270, 0.077391], [0.596729, 0.077391],
                       [0.699356, 0.034489], [0.808997, 0.018748], [0.914864, 0.038915],
                       [0.989913, 0.106454], [0.500000, 0.203352], [0.500000, 0.307009],
                       [0.500000, 0.409805], [0.500000, 0.515625], [0.376753, 0.587326],
                       [0.435909, 0.609345], [0.500000, 0.628106], [0.564090, 0.609345],
                       [0.623246, 0.587326], [0.131610, 0.216423], [0.196995, 0.178758],
                       [0.275698, 0.179852], [0.344479, 0.231733], [0.270791, 0.245099],
                       [0.192616, 0.244077], [0.655520, 0.231733], [0.724301, 0.179852],
                       [0.803005, 0.178758], [0.868389, 0.216423], [0.807383, 0.244077],
                       [0.729208, 0.245099], [0.264022, 0.780233], [0.350858, 0.745405],
                       [0.438731, 0.727388], [0.500000, 0.742578], [0.561268, 0.727388],
                       [0.649141, 0.745405], [0.735977, 0.780233], [0.652032, 0.864805],
                       [0.566594, 0.902192], [0.500000, 0.909281], [0.433405, 0.902192],
                       [0.347967, 0.864805], [0.300252, 0.784792], [0.437969, 0.778746],
                       [0.500000, 0.785343], [0.562030, 0.778746], [0.699747, 0.784792],
                       [0.563237, 0.824182], [0.500000, 0.831803], [0.436763, 0.824182]])

_MEAN_FACE_3D = np.array([[4.056931, -11.432347, 1.636229],   # 8 chin LL
                          [1.833492, -12.542305, 4.061275],   # 7 chin L
                          [0.0, -12.901019, 4.070434],        # 6 chin C
                          [-1.833492, -12.542305, 4.061275],  # 5 chin R
                          [-4.056931, -11.432347, 1.636229],  # 4 chin RR
                          [6.825897, 1.275284, 4.402142],     # 33 L eyebrow L
                          [1.330353, 1.636816, 6.903745],     # 29 L eyebrow R
                          [-1.330353, 1.636816, 6.903745],    # 34 R eyebrow L
                          [-6.825897, 1.275284, 4.402142],    # 38 R eyebrow R
                          [1.930245, -5.060977, 5.914376],    # 54 nose LL
                          [0.746313, -5.136947, 6.263227],    # 53 nose L
                          [0.0, -5.485328, 6.76343],          # 52 nose C
                          [-0.746313, -5.136947, 6.263227],   # 51 nose R
                          [-1.930245, -5.060977, 5.914376],   # 50 nose RR
                          [5.311432, 0.0, 3.987654],          # 13 L eye L
                          [1.78993, -0.091703, 4.413414],     # 17 L eye R
                          [-1.78993, -0.091703, 4.413414],    # 25 R eye L
                          [-5.311432, 0.0, 3.987654],         # 21 R eye R
                          [2.774015, -7.566103, 5.048531],    # 43 mouth L
                          [0.509714, -7.056507, 6.566167],    # 42 mouth top L
                          [0.0, -7.131772, 6.704956],         # 41 mouth top C
                          [-0.509714, -7.056507, 6.566167],   # 40 mouth top R
                          [-2.774015, -7.566103, 5.048531],   # 39 mouth R
                          [-0.589441, -8.443925, 6.109526],   # 46 mouth bottom R
                          [0.0, -8.601736, 6.097667],         # 45 mouth bottom C
                          [0.589441, -8.443925, 6.109526]])   # 44 mouth bottom L


def umeyama(source, destination, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Imported, and slightly adapted, directly from:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py


    Parameters
    ----------
    source: :class:`numpy.ndarray`
        (M, N) array source coordinates.
    destination: :class:`numpy.ndarray`
        (M, N) array destination coordinates.
    estimate_scale: bool
        Whether to estimate scaling factor.

    Returns
    -------
    :class:`numpy.ndarray`
        (N + 1, N + 1) The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    # pylint:disable=invalid-name,too-many-locals
    num = source.shape[0]
    dim = source.shape[1]

    # Compute mean of source and destination.
    src_mean = source.mean(axis=0)
    dst_mean = destination.mean(axis=0)

    # Subtract mean from source and destination.
    src_demean = source - src_mean
    dst_demean = destination - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    # T[:dim, :dim] = scale

    return T# , scale, dst_mean, src_mean

from .models import BiSeNet
def load_masking_model(device="cuda"):
    bisenet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained", "79999_iter.pth")

    mask_net = BiSeNet(19)
    mask_net.load_state_dict(torch.load(bisenet_path))
    mask_net.eval()
    mask_net.to(device)

    return mask_net




