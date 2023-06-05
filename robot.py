import socket
import select
import struct
import time
import os
import numpy as np
import utils
from simulation import vrep

class Robot(object):
    def __init__(self, obj_mesh_dir, num_obj, workspace_limits,
                 is_testing, test_preset_cases, test_preset_file):

        self.workspace_limits = workspace_limits
        # initial gripper position
        self.gripper_init_pos = (-0.5, 0, 0.5)
        self.gripper_init_pos_custom = None

        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                       [89.0, 161.0, 79.0], # green
                                       [237.0, 201.0, 72.0], # yellow
                                       [255.0, 87.0, 89.0], # red
                                        ])/255.0
        self.color_name = ['blue','green','yellow','red']
        self.color2place = {'blue':[1.675,-1.9,0.31],
                            'green':[1.325,-1.9,0.31],
                            'yellow':[2.025,-1.9,0.31],
                            'red':[0.975,-1.9,0.31]}
        self.color2workshop = {'blue':np.asarray([[1.525, 1.825], [-2.05, -1.75], [0.32, 0.33]]), # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
                            'green':np.asarray([[1.175, 1.475], [-2.05, -1.75], [0.32, 0.33]]),
                            'yellow':np.asarray([[1.875, 2.175], [-2.05, -1.75], [0.32, 0.33]]),
                            'red':np.asarray([[0.825, 1.125], [-2.05, -1.75], [0.32, 0.33]])
                               }

        # Read files in object mesh directory
        self.obj_mesh_dir = obj_mesh_dir
        self.num_obj = num_obj
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 4, :]

        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        self.is_testing = is_testing
        self.test_preset_cases = test_preset_cases
        self.test_preset_file = test_preset_file

        # Setup virtual camera in simulation
        self.setup_sim_camera()

        # Save the joints position, once we execute go_home() function, Franka go to the initial posture
        self.franka_joints_handle = []
        self.franka_initial_joint_positions = []
        self.franka_initial_joint_positions_custom = []
        for i in range(7):
            _, joint_handle = vrep.simxGetObjectHandle(self.sim_client, 'joint' + str(i+1), vrep.simx_opmode_blocking)
            self.franka_joints_handle.append(joint_handle)
        for joint_handle in self.franka_joints_handle:
            _, joint_position = vrep.simxGetJointPosition(self.sim_client, joint_handle, vrep.simx_opmode_blocking)
            self.franka_initial_joint_positions.append(joint_position)

        _, self.car_handle = vrep.simxGetObjectHandle(self.sim_client, 'Car_body', vrep.simx_opmode_blocking)
        sim_ret, self.left_motor_handle = vrep.simxGetObjectHandle(self.sim_client, 'LeftMotor', vrep.simx_opmode_blocking)
        sim_ret, self.right_motor_handle = vrep.simxGetObjectHandle(self.sim_client, 'RightMotor', vrep.simx_opmode_blocking)
        sim_ret, self.car_body_handle = vrep.simxGetObjectHandle(self.sim_client, 'Car_body_visual', vrep.simx_opmode_blocking)
        sim_ret, self.universal_handle = vrep.simxGetObjectHandle(self.sim_client, 'Universal_wheel',
                                                             vrep.simx_opmode_blocking)
        self.velocity_limit = 2.0


        # If testing, read object meshes and poses from test case file
        if self.is_testing and self.test_preset_cases:
            file = open(self.test_preset_file, 'r')
            file_content = file.readlines()
            self.test_obj_mesh_files = []
            self.test_obj_mesh_colors = []
            self.test_obj_positions = []
            self.test_obj_orientations = []
            for object_idx in range(self.num_obj):
                file_content_curr_object = file_content[object_idx].split()
                self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),3.21023232320e-01])
                self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
            file.close()
            self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

        # Add objects to simulation environment
        self.add_objects()



    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.1) * np.random.random_sample() + self.workspace_limits[0][0] + 0
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.1) * np.random.random_sample() + self.workspace_limits[1][0] + 0
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)
        self.prev_obj_positions = []
        self.obj_positions = []


    def restart_sim(self):
        # self.car_dynamic_disable()
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, self.gripper_init_pos, vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        # self.close_gripper()
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.7: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached


    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations


    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):

        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear


        return color_img, depth_img


    def close_gripper(self, asynch=False):
        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.045: # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            # print(gripper_joint_position)
            if new_gripper_joint_position >= gripper_joint_position:
                return gripper_fully_closed
            gripper_joint_position = new_gripper_joint_position
        gripper_fully_closed = True

        return gripper_fully_closed

    def open_gripper(self, asynch=False):

        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        while gripper_joint_position < 0.03: # Block until gripper is fully open
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)



    def move_to(self, tool_position, tool_orientation):
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.02))

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

    def car_dynamic_enable(self):
        _, self.car_handle = vrep.simxGetObjectHandle(self.sim_client, 'Car_body', vrep.simx_opmode_blocking)
        vrep.simxSetObjectIntParameter(self.sim_client, self.car_handle, vrep.sim_shapeintparam_static, 0, vrep.simx_opmode_blocking)

    def car_dynamic_disable(self):
        _, self.car_handle = vrep.simxGetObjectHandle(self.sim_client, 'Car_body', vrep.simx_opmode_blocking)
        vrep.simxSetObjectIntParameter(self.sim_client, self.car_handle, vrep.sim_shapeintparam_static, 1, vrep.simx_opmode_blocking)


    def go_home(self):

        self.move_to(self.gripper_init_pos,None)
        for i, joint_handle in enumerate(self.franka_joints_handle):
            vrep.simxSetJointPosition(self.sim_client, joint_handle, self.franka_initial_joint_positions[i], vrep.simx_opmode_blocking)

    def define_custom_home(self):
        _, self.gripper_init_pos_custom = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        for joint_handle in self.franka_joints_handle:
            _, joint_position = vrep.simxGetJointPosition(self.sim_client, joint_handle, vrep.simx_opmode_blocking)
            self.franka_initial_joint_positions_custom.append(joint_position)

    def go_costom_home(self):
        if len(self.franka_initial_joint_positions_custom):
            self.move_to(self.gripper_init_pos_custom,None)
            for i, joint_handle in enumerate(self.franka_joints_handle):
                vrep.simxSetJointPosition(self.sim_client, joint_handle, self.franka_initial_joint_positions_custom[i], vrep.simx_opmode_blocking)


    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))


        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

        # Move gripper to location above grasp target
        grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_direction[0]/move_step[0]))


        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        self.open_gripper()

        # Approach grasp target
        self.move_to(position, None)

        # Close gripper to grasp target
        gripper_full_closed = self.close_gripper()

        # Move gripper to location above grasp target
        self.move_to(location_above_grasp_target, None)


        # Check if grasp is successful
        gripper_full_closed = self.close_gripper()
        grasp_success = not gripper_full_closed

        # Move the grasped object elsewhere
        if grasp_success:
            object_positions = np.asarray(self.get_obj_positions())
            # print(object_positions)
            object_positions = object_positions[:,2]
            grasped_object_ind = np.argmax(object_positions)
            grasped_object_handle = self.object_handles[grasped_object_ind]
            vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.55, 1.5 + 0.01*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)

        return grasp_success

    def dev_color_match(self,color):
        color = color / 255.0
        distances = np.linalg.norm(self.color_space - color, axis=1)  # Calculate Euclidean distance
        index_of_smallest_distance = np.argmin(distances)  # Get the index of the smallest distance
        return self.color_name[index_of_smallest_distance]

    def dev_grasp_with_place(self, position, heightmap_rotation_angle, workspace_limits, color):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))


        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

        # Move gripper to location above grasp target
        grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_direction[0]/move_step[0]))


        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        self.open_gripper()

        # Approach grasp target
        self.move_to(position, None)

        # Close gripper to grasp target
        gripper_full_closed = self.close_gripper()

        # Move gripper to location above grasp target
        self.move_to(location_above_grasp_target, None)


        # Check if grasp is successful
        gripper_full_closed = self.close_gripper()
        grasp_success = not gripper_full_closed

        # Move the grasped object elsewhere
        if grasp_success:
            color_name = self.dev_color_match(color)
            drop_worklimits = self.color2workshop[color_name]
            drop_x = (drop_worklimits[0][1] - drop_worklimits[0][0] - 0.05) * np.random.random_sample() + \
                     drop_worklimits[0][0] + 0
            drop_y = (drop_worklimits[1][1] - drop_worklimits[1][0] - 0.1) * np.random.random_sample() + \
                     drop_worklimits[1][0] + 0
            object_position = [drop_x, drop_y, 0.31]
            #
            # self.place(object_position,0,workspace_limits)
            self.go_home()
            time.sleep(2)
            # self.car_dynamic_enable()
            self.car_move_to("tar_"+color_name)
            # target_place = self.color2place[color_name]
            self.place(object_position, 0)
            # self.go_home()
            time.sleep(2)
            self.car_move_to("sou")
            # self.car_dynamic_disable()


        return grasp_success


    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Adjust pushing point to be on tip of finger
        # position[2] = position[2] + 0.026

        # Compute pushing direction
        push_orientation = [1.0,0.0]
        push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

        # Move gripper to location above pushing point
        pushing_point_margin = 0.1
        location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_pushing_point
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is closed
        self.close_gripper()

        # Approach pushing point
        self.move_to(position, None)

        # Compute target location (push to the right)
        push_length = 0.1
        target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
        push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

        # Move in pushing direction towards target location
        self.move_to([target_x, target_y, position[2]], None)

        # Move gripper to location above grasp target
        self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

        push_success = True

        return push_success

    def place(self, position, heightmap_rotation_angle, workspace_limits=None):
        print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Avoid collision with floor
        # position[2] = max(position[2] + 0.04 + 0.02, workspace_limits[2][0] + 0.02)
        position[2]+=0.02

        # Move gripper to location above place target
        place_location_margin = 0.1
        sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_place_target = (position[0], position[1], position[2] + place_location_margin)
        self.move_to(location_above_place_target, None)

        sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
        if tool_rotation_angle - gripper_orientation[1] > 0:
            increment = 0.2
        else:
            increment = -0.2
        while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
            vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
            time.sleep(0.01)
            sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Approach place target
        self.define_custom_home()
        self.move_to(position, None)

        # Ensure gripper is open
        self.open_gripper()

        # Move gripper to location above place target
        self.move_to(location_above_place_target, None)

        place_success = True

        self.close_gripper()
        self.go_costom_home()

        return place_success

    def rotate_right(self, speed=2.0):
        speed = min(self.velocity_limit, speed)
        self._set_two_motor(speed, -speed)

    def rotate_left(self, speed=2.0):
        speed = min(self.velocity_limit, speed)
        self._set_two_motor(-speed, speed)

    def move_forward(self, speed=2.0):
        speed = min(self.velocity_limit, speed)
        speed = max(speed, 0.5)
        self._set_two_motor(speed, speed)

    def move_backward(self, speed=2.0):
        speed = min(self.velocity_limit, speed)
        speed = max(speed, 0.5)
        self._set_two_motor(-speed , -speed)

    def stand(self):
        self._set_two_motor(0, 0)

    def _set_two_motor(self, left: float, right: float):
        vrep.simxSetJointTargetVelocity(self.sim_client, self.left_motor_handle, left, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(self.sim_client, self.right_motor_handle, right, vrep.simx_opmode_oneshot)

    def move_to_tar(self, coord_x = 1.525):
        x_goal, y_goal = (coord_x, -1.10)
        _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
        x, y, z = position
        _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1, vrep.simx_opmode_blocking)
        x_uni, y_uni, z_uni = position_uni
        if y < -0.5:
            while abs(y_uni - y) > 0.002:
                if y_uni - y > 0:
                    self.rotate_right(abs(y_uni - y) * 5)
                else:
                    self.rotate_left(abs(y_uni - y) * 5)
                time.sleep(0.1)
                _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1,
                                                         vrep.simx_opmode_blocking)
                x, y, z = position
                _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1,
                                                             vrep.simx_opmode_blocking)
                x_uni, y_uni, z_uni = position_uni
            self.stand()
            time.sleep(0.1)
            while abs(x_goal - x) > 0.01:
                if x_goal - x > 0:
                    self.move_forward(abs(x_goal - x) * 5)
                else:
                    self.move_backward(abs(x_goal - x) * 5)

                _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1,
                                                         vrep.simx_opmode_blocking)
                x, y, z = position
            self.stand()
            time.sleep(0.1)
        else:
            while abs(x_uni - x) > 0.002:
                if x_uni - x > 0:
                    self.rotate_right(abs(x_uni - x) * 5)
                else:
                    self.rotate_left(abs(x_uni - x) * 5)
                time.sleep(0.1)
                _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
                x, y, z = position
                _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1,
                                                             vrep.simx_opmode_blocking)
                x_uni, y_uni, z_uni = position_uni
            self.stand()
            time.sleep(0.1)
            while abs(y_goal - y) > 0.01:
                if y_goal - y > 0:
                    self.move_backward(abs(y_goal - y) * 5)
                else:
                    self.move_forward(abs(y_goal - y) * 5)
                _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
                x, y, z = position

            self.stand()
            time.sleep(0.1)
            _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
            x, y, z = position
            _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1, vrep.simx_opmode_blocking)
            x_uni, y_uni, z_uni = position_uni
            while abs(y_uni - y) > 0.002:
                if y_uni - y > 0:
                    self.rotate_right(abs(y_uni - y) * 5)
                else:
                    self.rotate_left(abs(y_uni - y) * 5)
                time.sleep(0.1)
                _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
                x, y, z = position
                _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1,
                                                             vrep.simx_opmode_blocking)
                x_uni, y_uni, z_uni = position_uni
            self.stand()
            time.sleep(0.1)
            while abs(x_goal - x) > 0.01:
                if x_goal - x > 0:
                    self.move_forward(abs(x_goal - x) * 5)
                else:
                    self.move_backward(abs(x_goal - x) * 5)
                _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
                x, y, z = position

            self.stand()
            time.sleep(0.1)

    def move_to_sou(self):
        x_goal, y_goal = (-0.05, 0.0)
        _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
        x, y, z = position
        _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1, vrep.simx_opmode_blocking)
        x_uni, y_uni, z_uni = position_uni
        while abs(y_uni - y) > 0.002:
            if y_uni - y > 0:
                self.rotate_right(abs(y_uni - y) * 5)
            else:
                self.rotate_left(abs(y_uni - y) * 5)
            time.sleep(0.1)
            _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
            x, y, z = position
            _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1,
                                                         vrep.simx_opmode_blocking)
            x_uni, y_uni, z_uni = position_uni
        self.stand()
        time.sleep(0.1)
        while abs(x_goal - x) > 0.01:
            if x_goal - x > 0:
                self.move_forward(abs(x_goal - x) * 5)
            else:
                self.move_backward(abs(x_goal - x) * 5)

            _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
            x, y, z = position
        self.stand()
        time.sleep(0.1)
        _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
        x, y, z = position
        _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1, vrep.simx_opmode_blocking)
        x_uni, y_uni, z_uni = position_uni
        while abs(x_uni - x) > 0.002:
            if x_uni - x > 0:
                self.rotate_right(abs(x_uni - x) * 5)
            else:
                self.rotate_left(abs(x_uni - x) * 5)
            time.sleep(0.1)
            _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
            x, y, z = position
            _, position_uni = vrep.simxGetObjectPosition(self.sim_client, self.universal_handle, -1,
                                                         vrep.simx_opmode_blocking)
            x_uni, y_uni, z_uni = position_uni
        self.stand()
        time.sleep(0.1)
        while abs(y_goal - y) > 0.01:
            if y_goal - y > 0:
                self.move_backward(abs(y_goal - y) * 5)
            else:
                self.move_forward(abs(y_goal - y) * 5)

            _, position = vrep.simxGetObjectPosition(self.sim_client, self.car_body_handle, -1, vrep.simx_opmode_blocking)
            x, y, z = position
        self.stand()
        time.sleep(0.1)

    def car_move_to(self, position: str, velocity_limit: float = 2.0):
        """

        :param position: 点位字符串，可选：{"sou", “tar”（tar黄块中心）, "tar_red", "tar_green", "tar_blue", "tar_yellow"}
        :param velocity_limit: 小车限速
        """
        self.velocity_limit = velocity_limit
        if position == "sou":
            self.move_to_sou()
        elif position == "tar":
            self.move_to_tar(1.500)
        elif position == "tar_red":
            self.move_to_tar(0.975)
        elif position == "tar_green":
            self.move_to_tar(1.325)
        elif position == "tar_blue":
            self.move_to_tar(1.675)
        elif position == "tar_yellow":
            self.move_to_tar(2.025)
        else:
            raise Exception("Unknown Position Type : " + position)
