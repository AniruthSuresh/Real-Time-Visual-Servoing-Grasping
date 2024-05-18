"""
Setting the full robot environment and 
calling all the main functions 
"""

# updated ! 

import os
import shutil

from types import SimpleNamespace

from typing import Dict, List , Tuple

import numpy as np
import pybullet as p
from airobot import Robot
from airobot.arm.ur5e_pybullet import UR5ePybullet as UR5eArm
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils.common import clamp, euler2quat, euler2rot, quat2euler
from scipy.spatial.transform import Rotation as R


from utils.img_saver import ImageSaver
from utils.logger import logger

GRASP_TIME = 4 
REPEAT_ACTION = 10
DRIFT = 0.9
RENDERING = 0


def create_robot(gui: bool, check_realtime : bool , check_render :bool = False):
    """
    Create the robot and asserts with pybullet 
    """
    try:
        robot = Robot(
            "ur5e_2f140",
            # Have to keep openGL render off for the texture to work
            pb_cfg={"gui": gui, "realtime": check_realtime, "opengl_render": check_render},
        )
        assert p.isConnected(robot.pb_client.get_client_id())

    except:
       robot = Robot(
            "ur5e_2f140",
            pb_cfg={"gui": gui, "realtime": check_realtime, "opengl_render": check_render},
        )

    return robot



# all the def are method and self should the first argument in all of them 
class GYM_Robot:

    def __init__(
        self,
        init_cfg: dict,
        grasp_time=GRASP_TIME,
        gui=False,
        config: dict = {},
        controller_type="rtvs",
        record=True,
        flowdepth=True,
        ):
        
        self.gui = gui

        self.robot = create_robot(gui , check_realtime=False, check_render=False)

        # type hinting (:)
        self.cam: RGBDCameraPybullet = self.robot.cam
        self.arm: UR5eArm = self.robot.arm
        self.pb_client = self.robot.pb_client


        # set the configurations 
        self.set_configuration(init_cfg, grasp_time)
        
        # resets for the trail 
        self.reset_config()

        self.record_mode = record 
        self.flowdepth = flowdepth
        self.depth_noise = config.get("dnoise", 0)
        self.controller_type = controller_type

        self.set_RTVS_controller()


    def _configure_belt_motion(self, init_cfg):
        """
        Check whether the motion type is linear / circular 
        and call it accordingly
        """
        self.belt.vel = np.zeros(3)  # Initialize velocity as a zero vector
        
        if self.belt.motion_type == "linear":
            self._configure_linear_belt(init_cfg)

        elif self.belt.motion_type == "circle":
            self._configure_circular_belt(init_cfg)


    def _configure_linear_belt(self, init_cfg):
        """
        Assigns linear motion parameters 
        """

        self.belt.vel = np.array(init_cfg["obj_vel"])
        self.belt.vel[2] = 0
        self.belt.init_pos = np.array(init_cfg["obj_pos"])


    def _configure_circular_belt(self, init_cfg):
        """
        Assigns circular motion parameters 
        """

        self.belt.center = np.array(init_cfg["obj_center"])
        self.belt.radius = init_cfg["obj_radius"]
        self.belt.w = init_cfg["obj_w"]
        self.belt.init_pos = self.belt.center + self.belt.radius * np.array([1, 0, 0])


    def _configure_wall(self):
        """
        Just setting the wall in the env 
        """
        self.wall = SimpleNamespace()
        self.wall.init_pos = self.belt.init_pos + [0, 1, 0]
        self.wall.ori = np.deg2rad([90, 0, 0])
        self.wall.texture_name = "../data/texture.png"
        self.wall.scale = 1


    def _configure_table(self):
        """
        Just setting up things in the env 
        """
        self.table = SimpleNamespace()
        self.table.pos = np.array([0.5, 0, -5.4])
        self.table.ori = np.deg2rad([0, 0, 90])


    def _configure_box(self):
        """
        Configure the box and set it to green colour 
        with opacity = 1
        """
        self.box = SimpleNamespace()
        self.box.size = np.array([0.03, 0.06, 0.06])
        self.box.init_pos = np.array([*self.belt.init_pos[:2], DRIFT])
        self.box.init_ori = np.deg2rad([0, 0, 90])
        self.box.color = [0, 1, 0, 1]


    def set_configuration(self, init_cfg: dict, grasp_time:int=4):
        """
        Setting the orientation and position of the robot 
        in pybullet
        """

        self.step_dt = 1 / 250
        self.ground_lvl = 0.851
        p.setTimeStep(self.step_dt)


        self.repeat_action = REPEAT_ACTION # repeat it ten times 
        self._ee_pos_scale = self.step_dt * self.repeat_action

        self.cam_link_anchor_id = 22  #  or ('ur5_ee_link-gripper_base') link 10
        self.cam_ori = np.deg2rad([-105, 0, 0])
        self.cam_pos_delta = np.array([0, -0.2, 0.02])


        self.ee_home_pos = [0.5, -0.13, 0.9]
        self.ee_home_ori = np.deg2rad([-180, 0, -180])

        
        self.arm._home_position = [-0.0, -1.66, -1.92, -1.12, 1.57, 1.57]
        self.arm._home_position = self.arm.compute_ik(
            self.ee_home_pos, self.ee_home_ori
        )

        # constructs a rotation matrix from the Euler angles 
        # order of operation = xyz 
        self.rot_matrix = R.from_euler("xyz", self.cam_ori)
        
        self.ref_ee_ori = self.ee_home_ori[:]
        self.grasp_time = GRASP_TIME
        self.post_grasp_duration = 1


        self.ground = SimpleNamespace() # similar to dict but accessed using '.'
        self.ground.init_pos = [0, 0, self.ground_lvl - 0.005]
        self.ground.scale = 0.1

        self.belt = SimpleNamespace()
        self.belt.color = [0, 0, 0, 0]
        self.belt.scale = 0.1
        self.belt.motion_type = init_cfg["motion_type"]

    

        self._configure_belt_motion(init_cfg)
    

        self.belt.init_pos[2] = self.ground_lvl
        self.belt_setup_control(0)

        # Setting up the basic items in the env 
        self._configure_wall()
        self._configure_table()
        self._configure_box()


    def set_RTVS_controller(self):
        """
        Initialises the RTVS controller
        """

        logger.info(controller_type=self.controller_type)

        if self.controller_type == "rtvs":
            from controllers.rtvs import Rtvs, RTVSController

            self.controller = RTVSController(
                self.grasp_time,
                self.ee_home_pos,
                self.box.size,
                self.conveyor_level,
                self._ee_pos_scale,
                Rtvs("./dest.png", self.cam.get_cam_int()),
                self.rot_matrix,
                max_speed=0.7,
            )


    def belt_setup_control(self, t=None, dt=None):
        """

        Sets the linear and angular velocity of the belt so 
        effectively configuring the object position
        """
        if t is None:
            t = self.sim_time
        if dt is None:
            dt = self.step_dt * self.repeat_action

        if self.belt.motion_type == "circle":
            r = self.belt.radius
            w = self.belt.w
            self.belt.vel[0] = -w * r * np.sin(w * t)
            self.belt.vel[1] = w * r * np.cos(w * t)


    def get_pos(self, obj):
        """
        Check the type of obj and returns the corresponding 
        position after assigning its ID
        """
        if isinstance(obj, int):
            obj_id = obj
        elif hasattr(obj, "id"):
            obj_id = obj.id
        return self.pb_client.get_body_state(obj_id)[0]


    def debug_log(self):
        """
        Logs the initial positions, velocities, and other relevant information for debugging purposes.
        """
        ee_pos, _, _, ee_euler = self.arm.get_ee_pose()

        logger.info(init_belt_pose=self.get_pos(self.belt), belt_vel=self.belt.vel)
        logger.info(home_ee_pos=ee_pos, home_ee_euler=ee_euler)
        logger.info(home_jpos=self.arm.get_jpos())


    def reset_config(self):
        """
        Function to reset the simulation environment for a new trial.
        """

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, RENDERING)

        # resets the camera pos 
        p.resetDebugVisualizerCamera(
            cameraTargetPosition=[0, 0, 0],
            cameraDistance=2,
            cameraPitch=-40,
            cameraYaw=90,
        )


        change_friction = lambda id, lf=0, sf=0: p.changeDynamics(
            bodyUniqueId=id, linkIndex=-1, lateralFriction=lf, spinningFriction=sf
        )
        """
        Function to change the frictional prop of the object 


        id: The unique identifier of the object for which friction properties are to be changed.
        lf: The lateral friction coefficient.
        sf: The spinning friction coefficient
        """

        self.textures = {}

        def colour_or_texture(obj):
            # Ensure that the object has either "color" or "texture_name" attribute, but not both
            assert hasattr(obj, "color") ^ hasattr(obj, "texture_name")

            # Apply color if the object has a "color" attribute
            if hasattr(obj, "color"):
                p.changeVisualShape(obj.id, -1, rgbaColor=obj.color)
            # Apply texture if the object has a "texture_name" attribute
            else:
                # Check if the texture is already loaded
                if obj.texture_name not in self.textures:
                    # Load the texture and store its ID
                    tex_id = p.loadTexture(obj.texture_name)
                    assert tex_id >= 0
                    self.textures[obj.texture_name] = tex_id
                else:
                    # Retrieve the texture ID from the stored dictionary
                    tex_id = self.textures[obj.texture_name]
                
                # Apply the texture to the visual shape
                obj.texture_id = tex_id
                p.changeVisualShape(obj.id, -1, textureUniqueId=tex_id)


        self.arm.go_home(ignore_physics=True) # go back to initial pos 

        self.arm.eetool.open(ignore_physics=True) # open the gripper 


        self.ground.id = p.loadURDF(
            "plane.urdf", self.ground.init_pos, globalScaling=self.ground.scale
        )

        self.belt.id: int = p.loadURDF(
            "plane.urdf", self.belt.init_pos, globalScaling=self.belt.scale
        )


        self.wall.id: int = p.loadURDF(
            "plane.urdf",
            self.wall.init_pos,
            euler2quat(self.wall.ori),
            globalScaling=self.wall.scale,
        )
      

        # change the frictional motion of the belt 
        change_friction(self.belt.id, 2, 2)
      

        # apply colour or texture -> but not both -> xor !!
        colour_or_texture(self.belt)
        colour_or_texture(self.wall)


        # orientation is converted from euler to quat 
        self.box_id = self.robot.pb_client.load_geom(
            "box",
            size=(self.box.size / 2).tolist(),
            mass=1,
            base_pos=self.box.init_pos,
            rgba=self.box.color,
            base_ori=euler2quat(self.box.init_ori),
        )

        # loads the info for debugging 
        self.debug_log()

        # re-enable the rendering 
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


        # Renders and stores the image as prev for depth estimation 
        self.prev_rgb = self.render(for_video=False)[0]


        self.gripper_ori = 0
        self.belt_poses = []
        self.step_cnt = 0



    @property
    def conveyor_level(self):
        return self.ground_lvl

    @property
    def sim_time(self):
        return self.step_cnt * self.step_dt

    def perform_motion(self, action):
        """
        Applies the motion corresponding to the velocity obtained and then 
        appends the belt poses 
        """
        self.apply_action(action)
        self.belt_poses.append(self.get_pos(self.belt))


    def apply_action(self, action, use_belt=True):
        """
        Perform movements according to the velocity obatined using 
        DCEM and optical flow depth estimation
        """

        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()

        if action.size != 5:
            raise ValueError(
                "Action should be [d_x, d_y, d_z, angle, open/close gripper]."
            )
        
        # set the position 
        pos = self.ee_pos + action[:3] * self._ee_pos_scale
        pos[2] = max(pos[2], 0.01 + self.conveyor_level)


        int_pos = self.robot.arm.compute_ik(pos, ori=self.ee_home_ori) # inverse kinematics 
        gripper_ang = self._scale_gripper_angle(action[4])

        for step in range(self.repeat_action):
            self.arm.set_jpos(int_pos, wait=False, ignore_physics=(action[4] != 1))
            self.robot.arm.eetool.set_jpos(gripper_ang, wait=False)
            if use_belt:
                p.resetBaseVelocity(self.belt.id, self.belt.vel)
            self.robot.pb_client.stepSimulation()
            self.step_cnt += 1
        # logger.debug(action_target = pos, action_result = self.ee_pos, delta=(pos-self.ee_pos))


    def _scale_gripper_angle(self, command):
        """
        Convert the command in [-1, 1] to the actual gripper angle.
        command = -1 means open the gripper.
        command = 1 means close the gripper.

        Parameters:
            command (float): a value between -1 and 1.
                -1 means open the gripper.
                1 means close the gripper.

        Returns:
            float: the actual gripper angle
            corresponding to the command.
        """

        command = clamp(command, -1.0, 1.0)
        close_ang = self.robot.arm.eetool.gripper_close_angle
        open_ang = self.robot.arm.eetool.gripper_open_angle
        cmd_ang = (command + 1) / 2.0 * (close_ang - open_ang) + open_ang
        return cmd_ang


    @property # wrapper unit 
    def obj_pos(self):
        return self.pb_client.get_body_state(self.box_id)[0]

    @property
    def obj_pos_8(self):
        pos, quat = self.pb_client.get_body_state(self.box_id)[:2]
        euler_z = quat2euler(quat)[2]
        rotmat = euler2rot([0, 0, euler_z])

        points_base = [
            [1, 1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
        ]
        points_base = np.asarray(points_base) * self.box.size[2] / np.sqrt(2)
        points = points_base @ rotmat + pos
        points = np.asarray(sorted(points.tolist())).round(5)
        return np.asarray(sorted(points.tolist()))

    @property
    def ee_pos(self):
        return self.arm.get_ee_pose()[0]

    @property
    def cam_pos(self):
        ee_base_pos = p.getLinkState(self.arm.robot_id, self.cam_link_anchor_id)[0]
        return ee_base_pos + self.cam_pos_delta
    


    def render(self, get_rgb=True, get_depth=True, get_seg=True, for_video=True, noise=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Renders the setup and returns the rgb, depth, and seg images along with the camera position.
        
        Parameters:
            get_rgb (bool): Flag indicating whether to retrieve the RGB image. Default is True.
            get_depth (bool): Flag indicating whether to retrieve the depth image. Default is True.
            get_seg (bool): Flag indicating whether to retrieve the segmentation image. Default is True.
            for_video (bool): Flag indicating whether the render is for video. Default is True.
            noise (float or None): Optional parameter specifying the noise level to add to the depth image. Default is None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]: A tuple containing:
                - rgb (np.ndarray): The RGB image.
                - depth (np.ndarray): The depth image.
                - seg (np.ndarray): The segmentation image.
                - cam_eye (List[float]): The position of the camera.
        """

        if for_video:
            self.robot.cam.setup_camera(
                focus_pt=[0, 0, 0.7], dist=1.5, yaw=90, pitch=-40, roll=0
            )
        else:
            self.cam.set_cam_ext(pos=self.cam_pos, ori=self.cam_ori)


        cam_eye = self.cam_pos
        cam_dir = cam_eye + self.rot_matrix.apply([0, 0, 0.1])
        p.addUserDebugLine(cam_dir, cam_eye, [0, 1, 0], 3, 0.5)

        rgb, depth, seg = self.cam.get_images(
            get_rgb, get_depth, get_seg, shadow=0, lightDirection=[0, 0, 2]
        )
        
        if noise is not None:
            depth *= np.random.normal(loc=1, scale=noise, size=depth.shape)
        return rgb, depth, seg, cam_eye
    

    def save_img(self, rgb, t):
        """

        Function to save the images which can be converted to video 
        using the makevideo.sh
        """
        if not self.record_mode:
            return
        ImageSaver.save_rgb(rgb, t)



    def setup_image_directory(self):
        """
        Set up the directory for saving images.
        """
        if self.record_mode:
            shutil.rmtree("imgs", ignore_errors=True)
            os.makedirs("imgs", exist_ok=True)



    def simulation_run(self):
        """
        Main method which gets velocity and performs servoing and grasps 
        the moving object
        """

        state = {
            "obj_motion": {"motion_type": self.belt.motion_type},
            "cam_int": [],
            "cam_ext": [],
            "obj_pos": [],
            "obj_corners": [],
            "ee_pos": [],
            "joint_pos": [],
            "action": [],
            "t": [],
            "err": [],
            "cam_eye": [],
            "joint_vel": [],

            "images": {
                "rgb": [],
                "depth": [],
                "seg": [],
            },
            "grasp_time": self.grasp_time,
            "grasp_success": 0,
        }


        if self.belt.motion_type == "circle":
            state["obj_motion"].update(
                {
                    "radius": self.belt.radius,
                    "center": self.belt.center,
                    "w": self.belt.w,
                }
            )

        elif self.belt.motion_type == "linear":
            state["obj_motion"].update({"vel": self.belt.vel})



        def add_single_to_state(val, *args):
            """
            Adds a single parameter to the state dict 
            """
            if isinstance(val, list) or (
                isinstance(val, np.ndarray) and val.dtype == np.float64
            ):
                
                val = np.asarray(val, np.float32)
            nonlocal state
            list_val = state
            for arg in args:
                list_val = list_val[arg]
            list_val.append(val)


        def multi_add_to_state(*args):
            """

            Adds multiple entries by calling the add_single_to_state
            """
            for arg in args:
                add_single_to_state(*arg)


        # start the run here ! 
        logger.info("Run start", obj_pose=self.obj_pos)
        time_steps = 1 / (self.step_dt * self.repeat_action)

        
        self.setup_image_directory() # create a folder imgs



        total_sim_time = self.grasp_time + self.post_grasp_duration

        GRASPING = False
        GRASPING_SUCCESS = False

        # starts the servoing 
        t = 0

        while t < int(np.ceil(time_steps * total_sim_time)):
            rgb, depth, seg, cam_eye = self.render(
                for_video=False, noise=self.depth_noise
            )
 
 
            multi_add_to_state(
                (self.obj_pos, "obj_pos"),
                (self.obj_pos_8, "obj_corners"),
                (self.ee_pos, "ee_pos"),
                (self.arm.get_jpos(), "joint_pos"),
                (self.arm.get_jvel(), "joint_vel"),
                (t / time_steps, "t"),
                (cam_eye, "cam_eye"),
                (rgb, "images", "rgb"),
                (depth, "images", "depth"),
                (seg, "images", "seg"),
                (self.cam.get_cam_ext(), "cam_ext"),
                (self.cam.get_cam_int(), "cam_int"),
            )

            self.save_img(rgb, t)

            # create a new dict -> observation to store the img and current info
            status_new = {
                "cur_t": self.sim_time,
                "ee_pos": self.ee_pos, # ee = end - effector 
            }


            if self.controller_type == "rtvs":
                status_new["rgb_img"] = rgb
                status_new["depth_img"] = depth
                status_new["prev_rgb_img"] = self.prev_rgb


            # if flowdepth is True -> we use optical flow to calculate the depth 
            # so pop this depth image .
            if self.flowdepth: 
                status_new.pop("depth_img")


            # Get the velocity and the error using flownet !
            action, iou = self.controller.get_action(status_new)
            print(f"Velocity is  :{action} and IOU_curr : {iou}")


            if not GRASPING and self.controller.ready_to_grasp:
                logger.debug("Grasping start")
                GRASPING = True
                total_sim_time = self.sim_time + self.post_grasp_duration
                self.grasp_time = self.sim_time
                state["grasp_time"] = self.sim_time


            multi_add_to_state((action, "action"), (iou, "err"))
            logger.info(time=self.sim_time, action=action)
            logger.info(ee_pos=self.ee_pos, obj_pos=self.obj_pos)
           
            logger.info(
                dist=np.round(np.linalg.norm(self.ee_pos - self.obj_pos), 3),
                iou_err=iou,
            )


            self.belt_setup_control()

            self.perform_motion(action)
            
            self.prev_rgb = rgb

            if (
                GRASPING
                and not GRASPING_SUCCESS
                and ((self.obj_pos - self.belt.init_pos - self.box.size / 2)[2] > 0.02)
            ):
                GRASPING_SUCCESS = True
                logger.info("Grasping success")
            t += 1

        state["grasp_success"] = GRASPING_SUCCESS

        logger.info("Run end", ee_pos=self.ee_pos, obj_pose=self.obj_pos)
        return state



    def __del__(self):
        p.disconnect(self.pb_client.get_client_id())

