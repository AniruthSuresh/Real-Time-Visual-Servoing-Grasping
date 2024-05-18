"""

Basic functions for the controller actions 
"""

class Controller:
    def __init__(
        self,
        grasp_time,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        max_speed,
    ):
        self.ready_to_grasp = False
        self.real_grasp_time = None

        self.grasp_time = grasp_time
        self.post_grasp_dest = post_grasp_dest
        self.box_size = box_size
        self.max_speed = max_speed
        self.conveyor_level = conveyor_level
        self.ee_pos_scale = ee_pos_scale

    def _action_vel_to_target_pos(self, action_vel, ee_pos):
        return ee_pos + action_vel * self.ee_pos_scale

    def _target_pos_to_action_vel(self, tar_pos, ee_pos):
        return (tar_pos - ee_pos) / self.ee_pos_scale

    def get_action(self, obsevations:dict):
        raise NotImplementedError

