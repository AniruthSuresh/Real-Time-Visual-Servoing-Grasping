"""
Main file implementing the servoing after setting up the env
"""

import argparse

from typing import Dict, List
import numpy as np

from gym_setup import GYM_Robot


np.set_string_function(
    lambda x: repr(np.round(x, 4))
    .replace("(", "")
    .replace(")", "")
    .replace("array", "")
    .replace("       ", " "),
    repr=False,
)


def simulate(init_cfg: dict, gui, controller, record: bool, flowdepth: bool):
    """
    Calls the gym setup and sets up the full robot env
    """
    env = GYM_Robot(init_cfg, gui=gui, controller_type=controller, record=record, flowdepth=flowdepth)
    return env.simulation_run() # calls the main method here ! 



def get_parser():
    """
    function creates an argument parser for parsing command-line arguments 
    """
    parser = argparse.ArgumentParser()

    # Add the controller type using -c or --c and has to be a string.
    # If not provided, default value is "rtvs".
    parser.add_argument("-c","--controller",type=str,default="rtvs",help="controller",)

    parser.add_argument("--dnoise", type=float, default=0, help="depth noise") # add env noise 

    parser.add_argument("--random", action="store_true")

    parser.add_argument("--gui", action="store_true", help="show gui")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="no gui")

    parser.add_argument("--circle", action="store_true", help="move in circle")

    parser.add_argument("--record", action="store_true", help="save imgs")
    parser.add_argument("--no-record", dest="record", action="store_false")


    parser.add_argument("--flowdepth", action="store_true", help="use flow_depth")

    # Set default values for gui and record
    parser.set_defaults(gui=False, record=True)

    return parser


def set_initial_position() -> Dict[str, List[float]]:
    """
    Generates the initial configuration dictionary for setting the robot's initial position.
    """
    init_cfg = {
        "motion_type": "linear",
        "obj_pos": [0.45, -0.05, 0.851],
        "obj_vel": [-0.01, 0.03, 0],
    }
    return init_cfg


def update_random_config(init_cfg: dict, args):
    """
    Updates the initial config to random type if the argument is true
    """

    if args.random and args.circle:
        init_cfg.update(
            {
                "obj_w": np.random.uniform(1, 5),
                "obj_radius": np.random.uniform(0.01, 0.05),
            }
        )
    elif args.random:
        init_cfg.update(
            {
                "obj_vel": np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]),
            }
        )
    return init_cfg



def update_to_circular_motion(init_cfg: dict, args):
    """
    Updates the initial config to circular type if the argument is true
    """
    if args.circle:
        init_cfg = {
            "motion_type": "circle",
            "obj_center": [0.45, -0.05, 0.851],
            "obj_w": 3,
            "obj_radius": 0.03,
        }
    return init_cfg



def main()-> None:

    # get all the arguments 
    parser = get_parser()
    args = parser.parse_args()


    # initial configuration of the robot -> moves with this velocity and at this position
    init_cfg = set_initial_position()

    # check for circular motion 
    init_cfg = update_to_circular_motion(init_cfg, args)

    # check for random motion
    init_cfg = update_random_config(init_cfg, args)


    simulation_result = simulate(
        init_cfg=init_cfg,
        gui=args.gui,
        controller=args.controller,
        record=args.record,
        flowdepth=args.flowdepth,
    )

    return simulation_result


if __name__ == "__main__":
    main()


