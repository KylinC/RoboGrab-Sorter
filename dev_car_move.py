#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import time
import cv2
import os
import random
from dev_robot import Robot
import utils


# User options (change me)
# --------------- Setup options ---------------
obj_mesh_dir = os.path.abspath('objects/blocks')
num_obj = 10
random_seed = 1234
workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [0.30, 0.9]]) # Cols: min max, Rows: y x z (define workspace limits in robot coordinates)
# ---------------------------------------------

# Set random seed
np.random.seed(random_seed)

# Initialize robot simulation
robot = Robot(obj_mesh_dir, num_obj, workspace_limits,
              True, False, None)

# robot.car_dynamic_enable()

print(1)

robot.car_move_to("tar_green")
robot.car_move_to("sou")
