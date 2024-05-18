"""
Checks for the condition for Grasping and performs Grasping
if satisfied
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger import logger

from ..base_controller import Controller
from .RTVS import Rtvs


THRESHOLD_GRASP = 0.6 

READY_GRASP = 0.9

class RTVSController(Controller):

    def __init__(
        self,
        grasp_time: float,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        rtvs: Rtvs, # dest and curr cam image 
        rot_matrix: R,
        max_speed=0.5,
    ):
        
        # any initialization logic defined in the Controller class is
        # executed before the specific initialization logic of the RTVSController class.

        super().__init__(
            grasp_time,
            post_grasp_dest,
            box_size,
            conveyor_level,
            ee_pos_scale,
            max_speed,
        )
        self.rtvs = rtvs
        self.rot_matrix = rot_matrix


    def normalised_vel_gnd_frame(self, rgb_img, depth_img, prev_rgb_img):
        """
        
        After getting the velocity , we convert this to ground frame from camera frame 
        and then normalise the velocity to avoid jerky motion of the end effector 
        """
        ee_vel_cam, err = self.rtvs.get_vel(
            rgb_img, depth=depth_img, pre_img_src=prev_rgb_img
        )

        ee_vel_cam = ee_vel_cam[:3]
        # The linear velocity obtained in the camera frame (ee_vel_cam) is 
        # transformed to the ground truth frame (ee_vel_gt) using the rotation matrix 

        # camera to ground frame 
        ee_vel_gt = self.rot_matrix.apply(ee_vel_cam)

        # normalised to stay in limits 
        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt))
        vel = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        )

        if err > READY_GRASP:
            self.ready_to_grasp = True

        logger.debug(
            "controller (rtvs frame):",
            pred_vel=vel,
            pred_speed=np.linalg.norm(vel),
            photo_err=err,
        )
        
        return vel
    

    def open_phase(self, action, cur_t, rgb_img, depth_img, prev_rgb_img, ee_pos):
        """
        Executes the actions for the open phase of grasping.
        """
        if cur_t <= self.grasp_time and not self.ready_to_grasp:
            action[4] = -1  # Open the gripper
            action[:3] = self.normalised_vel_gnd_frame(rgb_img, depth_img, prev_rgb_img)

            if cur_t <= THRESHOLD_GRASP * self.grasp_time:
                tpos = self._action_vel_to_target_pos(action[:3], ee_pos)
                action[2] = self._target_pos_to_action_vel(tpos, ee_pos)[2]


    def close_phase(self, action, cur_t, rgb_img, depth_img, prev_rgb_img, ee_pos):
        """
        Executes the actions for the close phase of grasping.
        """
        if cur_t > self.grasp_time:
            action[4] = 1  # Close the gripper

            if self.real_grasp_time is None:
                self.real_grasp_time = cur_t

           # end effector position is determined using the normalised_vel_gnd_frame 
            #function to ensure a smooth transition after grasping
            if cur_t <= self.real_grasp_time + 0.5:
                action[:3] = self.normalised_vel_gnd_frame(rgb_img, depth_img, prev_rgb_img)

            #end effector position is set to a fixed position 
            #above the grasped object to prevent collisions during lifting
                
            elif cur_t <= self.real_grasp_time + 1.0:
                action[:3] = [0, 0, 0.5]

            #destination position after grasping.
            else:
                action[:3] = self.post_grasp_dest - ee_pos


    def get_action(self, status_new: dict): 
        """
        Determines the action to be performed based on the current status.
        """
        ee_pos = status_new["ee_pos"]
        cur_t = status_new["cur_t"]
        rgb_img = status_new["rgb_img"] 
        depth_img = status_new.get("depth_img", None)
        prev_rgb_img = status_new.get("prev_rgb_img", None)

        action = np.zeros(5)

        self.open_phase(action, cur_t, rgb_img, depth_img, prev_rgb_img, ee_pos)
        self.close_phase(action, cur_t, rgb_img, depth_img, prev_rgb_img, ee_pos)

        return action, self.rtvs.get_iou(rgb_img)