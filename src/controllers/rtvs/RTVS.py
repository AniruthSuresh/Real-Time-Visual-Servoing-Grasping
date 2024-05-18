"""

Computes the velocity using the current and prev image using optical flow 
and returns this velocity vector to the controller
"""

from typing import Tuple


import numpy as np
import torch

from utils.img_saver import ImageSaver
from utils.logger import logger

from .Masking import BaseRtvs
from .DCEM_Sampling import Model
from .utils.flow_utils import flow2img



class Rtvs(BaseRtvs): 

    def __init__(
        self,
        img_goal: np.ndarray, #destination pic 
        cam_k: np.ndarray, # current pic 
        ct=1,
        horizon=20,
        LR=0.005,
        iterations=10,
    ):
        self.vs_lstm = Model().to(device="cuda:0")
        super().__init__(img_goal, cam_k, ct, horizon, LR, iterations)


    def compute_depth(self , flow_utils, img_src, pre_img_src=None, depth=None, ct=None) -> np.ndarray:
        """
        Computes the depth based on optical flow and optionally provided depth information.

        Parameters:
            flow_utils: Object providing utilities for flow calculations.
            img_src: Current RGB camera image.
            pre_img_src: Previous RGB camera image (used for depth estimation).
            depth: Depth information (optional).
            ct: Downsampling factor.

        Returns:
            final_depth: Computed depth values.
        """

        if depth is None: 
            flow_depth_proxy = (
                flow_utils.flow_calculate(img_src, pre_img_src).astype("float64")
            )[::ct, ::ct]
            flow_depth = np.linalg.norm(flow_depth_proxy, axis=2).astype("float64")
            final_depth = 0.1 / ((1 + np.exp(-1 / flow_depth)) - 0.5)
        else:
            final_depth = (depth[::ct, ::ct] + 1) / 10

        return final_depth


    def preprocess_masks(self, img_src: np.ndarray, ct: float) -> Tuple[np.ndarray, float]:
        """
        Preprocesses the object mask and computes the IoU score.
        """
        lower_bound = (50, 100, 100)
        upper_bound = (70, 255, 255)
        obj_mask = self.detect_mask(img_src, (lower_bound, upper_bound))
        iou_score = self.get_iou(img_src)
        obj_mask = obj_mask[::ct, ::ct]
        return obj_mask, iou_score


    def numpy_to_tensor(self, array, device="cuda:0"):
        """
        Convert a numpy array to a PyTorch tensor and move it to the specified device.
        """
        tensor = torch.tensor(array, dtype=torch.float32)
        tensor = tensor.to(device)
        return tensor
    
    def prepare_input_and_apply_pooling(f12, vs_lstm):
        """

        Prepare the input tensor for the vs_lstm model by rearranging dimensions and adding a batch dimension,
        then apply pooling operation to the input tensor using the vs_lstm model.

        """
        # Rearrange dimensions
        f12_permuted = f12.permute(2, 0, 1)
        # Add batch dimension
        f12_prepared = f12_permuted.unsqueeze(dim=0)
        # Apply pooling operation
        f12_pooled = vs_lstm.pooling(f12_prepared)
        return f12_pooled


    def get_vel(self, img_src, pre_img_src=None, depth=None)-> Tuple[np.ndarray, float]:
        """
        Paramters:
            img_src = current RGB camera image
            prev_img_src = previous RGB camera image
            (to be used for depth estimation using flowdepth)
            
            Returns the velocity for the controller 
        """

        img_goal = self.img_goal
        flow_utils = self.flow_utils
        vs_lstm = self.vs_lstm
        loss_fn = self.loss_fn
        optimiser = self.optimiser
        img_src = img_src
        ct = self.ct


        if depth is not None:
            depth = depth

        if pre_img_src is not None:
            pre_img_src = pre_img_src


        self.cnt = 0 if not hasattr(self, "cnt") else self.cnt + 1

        # get the masked and iou score -> interest only on a specific region 
        obj_mask, iou_score = self.preprocess_masks(img_src, ct)


        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        f12 = f12 * obj_mask


        ImageSaver.save_flow_img(flow2img(f12), self.cnt)


        final_depth = self.compute_depth(flow_utils, img_src, pre_img_src, depth, ct)


        vel, Lsx, Lsy = get_interaction_matrix(final_depth, ct, self.cam_k)


        Lsx = Lsx * obj_mask
        Lsy = Lsy * obj_mask

        Lsx = self.numpy_to_tensor(Lsx, device="cuda:0")
        Lsy = self.numpy_to_tensor(Lsy, device="cuda:0")
        f12 = self.numpy_to_tensor(f12, device="cuda:0")


        # apply pooling before passing through the neural network 
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))
 
        for itr in range(self.iterations):
            vs_lstm.v_interm = []
            vs_lstm.f_interm = []
            vs_lstm.mean_interm = []

            vs_lstm.zero_grad()
            f_hat = vs_lstm.get_pred_flow(vel, Lsx, Lsy, self.horizon, f12)
            loss = loss_fn(f_hat, f12) # computes the loss 

            logger.debug(rtvs_mse=loss.item() ** 0.5, rtvs_itr=itr)
            loss.backward(retain_graph=True)
            optimiser.step()

        # Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        with torch.no_grad():
            f_hat = vs_lstm.get_pred_flow(
                vel, Lsx, Lsy, -self.horizon, f12.to(torch.device("cuda:0"))
            )

        vel = vs_lstm.v_interm[0].detach().cpu().numpy() # get the vel corresponding to the least error-> to the controller

        logger.info(RAW_RTVS_VELOCITY=vel)

        return vel, iou_score


def extract_camera_parameters(cam_k)-> Tuple[float, float, float, float]:
    """

    Function to extracget_action
    |  0  fy   cy |
    |  0   0    1 |

    fx: Focal length in the x-direction (in pixels)
    fy: Focal length in the y-direction (in pixels)
    cx: Principal point offset along the x-axis (in pixels)
    cy: Principal point offset along the y-axis (in pixels)
    """

    kx = cam_k[0, 0]
    ky = cam_k[1, 1]
    Cx = cam_k[0, 2]
    Cy = cam_k[1, 2]
    return kx, ky, Cx, Cy


def get_interaction_matrix(d1: np.ndarray, ct: float, cam_k: np.ndarray) -> Tuple[None, np.ndarray, np.ndarray]:
    """
    Computes the interaction matrix based on the given depth map, camera parameters, and time constant.

    Args:
        d1 (np.ndarray): Depth map.
        ct : Downsampling factor . 
        cam_k (np.ndarray): Camera intrinsic matrix.

    Returns:
        Tuple[None, np.ndarray, np.ndarray]: A tuple containing None (no specific return value), Lsx, and Lsy matrices.
    """
    kx, ky, Cx, Cy = extract_camera_parameters(cam_k)
    
    med = np.median(d1)
    i, j = np.indices(d1.shape)
    
    xyz = np.zeros((d1.shape[0], d1.shape[1], 3))


    for k in range(3):
        coeff1 = 0.5 * (k - 1) * (k - 2)
        coeff2 = -k * (k - 2)
        coeff3 = 0.5 * k * (k - 1)
        d1_med = np.where(d1 == 0, med, d1)
        
        xyz[:, :, k] = (coeff1 * (ct * j - Cx) / kx +
                        coeff2 * (ct * i - Cy) / ky +
                        coeff3 * d1_med)
    
    Lsx = np.zeros((d1.shape[0], d1.shape[1], 6))
    Lsy = np.zeros((d1.shape[0], d1.shape[1], 6))
    
    # For x 
    Lsx[:, :, 0] = -1 / xyz[:, :, 2] # -1/z
    # 1 is just 0 in the matrix (include in the defintion )
    Lsx[:, :, 2] = xyz[:, :, 0] / xyz[:, :, 2] # x/z
    Lsx[:, :, 3] = xyz[:, :, 0] * xyz[:, :, 1]# xy
    Lsx[:, :, 4] = -(1 + xyz[:, :, 0] ** 2) # -(1+x2)
    Lsx[:, :, 5] = xyz[:, :, 1] # y


    # For y 
    # first is 0 -> included in the definition 
    Lsy[:, :, 1] = -1 / xyz[:, :, 2] # -1/z
    Lsy[:, :, 2] = xyz[:, :, 1] / xyz[:, :, 2] # y/Z
    Lsy[:, :, 3] = 1 + xyz[:, :, 1] ** 2 # (1+y2)
    Lsy[:, :, 4] = -xyz[:, :, 0] * xyz[:, :, 1] #-xy
    Lsy[:, :, 5] = -xyz[:, :, 0] # -x
    
    return None, Lsx, Lsy


