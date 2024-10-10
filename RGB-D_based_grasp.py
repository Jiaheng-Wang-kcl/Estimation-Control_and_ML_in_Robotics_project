import math
import random
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pinocchio as pin
import pybullet as p
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from utils.camera import Camera, eulerAnglesToRotationMatrix, getTransfMat
import utils.tool as tool
from ggcnn.ggcnn import GGCNNNet, drawGrasps, drawRect, getGraspDepth
import skimage.transform as skt
import scipy.stats as ss
import cv2


# IMAGEWIDTH = 640
# IMAGEHEIGHT = 480
IMAGEWIDTH = 320
IMAGEHEIGHT = 320

# IMAGEWIDTH = 80
# IMAGEHEIGHT = 56

nearPlane = 0.01
farPlane = 10

fov = 60    # Vertical field of view, image height tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT
FINGER_L1 = 0.015  # Length of the robotic finger in meters
FINGER_L2 = 0.005  # Unit: meters

def adjust_grasp_angle(grasp_x, grasp_y, grasp_angle, robot_id):
    # Get the position of the robot base
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    
    # Take the x and y coordinates of the base
    robot_base_x = base_pos[0]
    robot_base_y = base_pos[1]
    
    # Calculate the azimuth angle of the grasp point relative to the robot
    angle_to_robot = math.atan2(grasp_y - robot_base_y, grasp_x - robot_base_x)
    
    # If the grasp angle faces inward toward the robot (angle < 90 degrees), reverse the grasp angle
    if abs(grasp_angle - angle_to_robot) < math.pi / 2:
        grasp_angle += math.pi
    
    # Ensure the grasp angle is within a reasonable range
    grasp_angle = grasp_angle % (2 * math.pi)
    
    return grasp_angle

# Draw xyz axis lines in PyBullet
def draw_grasp_point(sim, grasp_x, grasp_y, grasp_z, line_length=1):
    # Color definitions: red for x-axis, green for y-axis, blue for z-axis
    line_color_x = [1, 0, 0]  # Red
    line_color_y = [0, 1, 0]  # Green
    line_color_z = [0, 0, 1]  # Blue

    # Endpoint coordinates: extend line_length distance along each axis from the grasp point
    grasp_point = [grasp_x, grasp_y, grasp_z]
    
    # X-axis line
    end_x = [grasp_x + line_length, grasp_y, grasp_z]
    sim.pybullet_client.addUserDebugLine(grasp_point, end_x, line_color_x, lineWidth=2)
    
    # Y-axis line
    end_y = [grasp_x, grasp_y + line_length, grasp_z]
    sim.pybullet_client.addUserDebugLine(grasp_point, end_y, line_color_y, lineWidth=2)
    
    # Z-axis line
    end_z = [grasp_x, grasp_y, grasp_z + line_length]
    sim.pybullet_client.addUserDebugLine(grasp_point, end_z, line_color_z, lineWidth=2)

def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic' interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image, output_shape, order=skt_interp_map[interp], anti_aliasing=False, mode="constant")

class RGBD_Grasp_Env(pb.SimInterface):
    def __init__(self, conf_file_name: str, conf_file_path_ext: str = None):
        """
        Initialize the RGBD_Grasp_Env environment, inherited from SimInterface
        """
        super().__init__(conf_file_name, conf_file_path_ext)
        self.camera = Camera(fov = fov, length = 0.7, height = IMAGEHEIGHT, width = IMAGEWIDTH)  # Set camera parameters using existing Camera class
        
        self.movecamera(0, -0.7, 1.0)
        
        self.ggcnn = GGCNNNet('ggcnn/ckpt/epoch_0213_acc_0.6374.pth', device="cpu")  # Initialize ggcnn

        self.projectionMatrix = self.pybullet_client.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )
        self.armId = self.bot[0].bot_pybullet
        self.dyn_model = None

        # Read object paths
        self.path = "myModel/objs"
        self.urdfs_list = []
        list_file = os.path.join(self.path, 'list.txt')
        with open(list_file, 'r') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                self.urdfs_list.append(os.path.join(self.path, line.strip() + '.urdf'))

    def keepPosTorque(self, dyn_model, keep_joints_angle):
        if self.dyn_model is None:
            self.dyn_model = dyn_model
            self.num_joints = 9    
        q_current = p.getJointStates(self.armId, range(self.num_joints))
        q_current_positions = np.array([state[0] for state in q_current[:self.num_joints]])
        pin.forwardKinematics(self.dyn_model.pin_model, self.dyn_model.pin_data, q_current_positions)

        for joint_id in range(self.num_joints):
            target_angle = keep_joints_angle[joint_id]
            
            self.pybullet_client.setJointMotorControl2(
                bodyUniqueId=self.armId,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle
            )
            
        self.pybullet_client.stepSimulation()

    def loadObjsInURDF(self, idx, num):
        """
        Load multiple obj objects in URDF format
        
        num: Number of objects to load
        idx: Starting ID
            If idx is negative, randomly load num objects
            If idx is non-negative, load num objects starting from id
        """
        assert idx >= 0 and idx < len(self.urdfs_list), "Index out of range"
        self.num_urdf = num
    
        # Get object file list
        if (idx + self.num_urdf - 1) > (len(self.urdfs_list) - 1):
            self.urdfs_filename = self.urdfs_list[idx:]
            self.urdfs_filename += self.urdfs_list[:2 * self.num_urdf - len(self.urdfs_list) + idx]
            self.objs_id = list(range(idx, len(self.urdfs_list)))
            self.objs_id += list(range(self.num_urdf - len(self.urdfs_list) + idx))
        else:
            self.urdfs_filename = self.urdfs_list[idx:idx + self.num_urdf]
            self.objs_id = list(range(idx, idx + self.num_urdf))
    
        print('Loaded object IDs = \n', self.objs_id)
    
        # Initialize storage list for objects
        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
    
        # base_x = self.camera_pos[0]
        # base_y = self.camera_pos[1] 
        # for i in range(self.num_urdf):
        #     pos = 0.2
        #     basePosition = [base_x + random.uniform(-pos, pos), base_y + random.uniform(-pos, pos), random.uniform(0.2, 0.3)]
    
        #     baseEuler = [random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)]
        #     baseOrientation = self.pybullet_client.getQuaternionFromEuler(baseEuler)
    
        #     urdf_id = self.pybullet_client.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)
            
        #     inf = self.pybullet_client.getVisualShapeData(urdf_id)[0]
        #     self.urdfs_id.append(urdf_id)
        #     self.urdfs_xyz.append(inf[5])  # Position
        #     self.urdfs_scale.append(inf[3][0])  # Scale
    
        #     self.set_gripper_and_nearby_collision(urdf_id)

        # Generate center values and ranges for direction and radius
        direction_center = -math.pi / 2  # Center direction angle (radians)
        direction_range = math.pi / 3    # Random offset range for direction angle (radians)

        radius_center = 0.7  # Center radius
        radius_range = 0.07   # Random offset range for radius

        for i in range(self.num_urdf):
            # Randomly generate direction angle and radius
            direction = direction_center + random.uniform(-direction_range, direction_range)
            radius = radius_center + random.uniform(-radius_range, radius_range)
            
            # Convert polar coordinates to Cartesian coordinates
            basePosition = [radius * math.cos(direction),
                            radius * math.sin(direction),
                            random.uniform(0.2, 0.3)]
            
            # Random orientation
            baseEuler = [random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)]
            baseOrientation = self.pybullet_client.getQuaternionFromEuler(baseEuler)
            
            # Load the object
            urdf_id = self.pybullet_client.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)
            
            # Retrieve object information
            inf = self.pybullet_client.getVisualShapeData(urdf_id)[0]
            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5])  # Position
            self.urdfs_scale.append(inf[3][0])  # Scaling factor
            
            # Set collision filtering, only for the gripper and nearby joints
            self.set_gripper_and_nearby_collision(urdf_id)

    
        print("Objects loaded successfully")
    
    def predict_grasp(self):
        camera_depth = self.renderCameraDepthImage()

        row, col, grasp_angle, grasp_width_pixels = self.ggcnn.predict(camera_depth, input_size=300)

        grasp_width = self.camera.pixels_TO_length(grasp_width_pixels, camera_depth[row, col])

        grasp_x, grasp_y, grasp_z = self.camera.img2world([col, row], camera_depth[row, col])
        
        adjusted_grasp_angle = adjust_grasp_angle(grasp_x, grasp_y, grasp_angle, self.armId)
        grasp_angle = adjusted_grasp_angle

        finger_l1_pixels = self.camera.length_TO_pixels(FINGER_L1, camera_depth[row, col])
        finger_l2_pixels = self.camera.length_TO_pixels(FINGER_L2, camera_depth[row, col])
        grasp_depth = getGraspDepth(camera_depth, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
        grasp_z = max(self.camera.length - grasp_depth, 0)

        print(f"Grasp position: x={grasp_x}, y={grasp_y}, z={grasp_z}, angle={grasp_angle}, width={grasp_width}")

        self.visualize_grasp(camera_depth, row, col, grasp_angle, grasp_width_pixels)

        return grasp_x, grasp_y, grasp_z, grasp_angle, grasp_width

    def visualize_grasp(self, camera_depth, row, col, grasp_angle, grasp_width_pixels):
        """
        Visualize the grasp configuration
        :param camera_depth: Depth map
        :param row: Row coordinate of the grasp point
        :param col: Column coordinate of the grasp point
        :param grasp_angle: Grasp angle
        :param grasp_width_pixels: Grasp width in pixels
        """
        im_rgb = tool.depth2Gray3(camera_depth)
        
        im_grasp = drawGrasps(im_rgb, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')
        
        cv2.imshow('Grasp Visualization', im_grasp)
        cv2.waitKey(1500)  # Wait 30ms to refresh the image
        cv2.destroyWindow('Grasp Visualization') 

    def set_gripper_and_nearby_collision(self, urdf_id):
        """
        Set collision for the gripper and its nearby joints with the object
        urdf_id: ID of the object
        """
        links_to_collide = ["panda_hand", "panda_link6", "panda_link7", "panda_leftfinger", "panda_rightfinger"]
    
        robot_id = self.armId  
    
        for link_name in links_to_collide:
            link_index = self.get_link_index_by_name(robot_id, link_name)
            if link_index is not None:
                self.pybullet_client.setCollisionFilterPair(urdf_id, robot_id, -1, link_index, 1)
       
    def get_link_index_by_name(self, robot_id, link_name):
        """
        Get the link index corresponding to the link name in URDF
        """
        num_joints = self.pybullet_client.getNumJoints(robot_id)
        for joint_index in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(robot_id, joint_index)
            if joint_info[12].decode("utf-8") == link_name:  # joint_info[12] is the link name
                return joint_index
        return None
    
    def movecamera(self, x, y, z=0.7, pitch=math.pi, yaw=0, roll=0):
        """
        Move the camera to the specified position and set viewMatrix
        """
        self.camera_pos = np.array([x, y, z])
        self.viewMatrix = self.pybullet_client.computeViewMatrix(
            [x, y, z], [x, y, 0], [0, 1, 0]
        )
        self.camera_target = [x, y, 0]

        self.camera.fov = fov
        self.camera.length = z
        rotMat = eulerAnglesToRotationMatrix([pitch, yaw, roll])
        self.camera.transMat = getTransfMat([x, y, z], rotMat)

    def renderCameraDepthImage(self):
        """
        Render a depth image for grasp configuration computation
        """
        t = time.time()
        img_camera = self.pybullet_client.getCameraImage(
            IMAGEWIDTH, IMAGEHEIGHT, 
            self.viewMatrix, self.projectionMatrix, 
            renderer=self.pybullet_client.ER_TINY_RENDERER
        )

        print("t1:",time.time() - t)
        t = time.time()
        w = img_camera[0]
        h = img_camera[1]
        dep = img_camera[3]

        depth = np.reshape(dep, (h, w))
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))
        print("t2:",time.time() - t)
        return im_depthCamera
    
    def gaussian_noise(self, im_depth):
        """
        Add Gaussian noise to the image, referencing dex-net code
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = 0.002
        gaussian_process_scaling_factor = 8.0

        im_height, im_width = im_depth.shape
        
        gp_rescale_factor = gaussian_process_scaling_factor
        gp_sample_height = int(im_height / gp_rescale_factor)
        gp_sample_width = int(im_width / gp_rescale_factor)
        gp_num_pix = gp_sample_height * gp_sample_width
        gp_sigma = gaussian_process_sigma
        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)
        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")
        im_depth += gp_noise

        return im_depth

    def add_noise(self, img):
        """
        Add Gaussian noise
        """
        img = self.gaussian_noise(img)
        return img

def main():
    conf_file_name = "panda_grasp_config.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = RGBD_Grasp_Env(conf_file_name, conf_file_path_ext = cur_dir)
    
    sim.loadObjsInURDF(0, 3)

    camera_depth = sim.renderCameraDepthImage()
    # camera_depth = sim.add_noise(camera_depth)

    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)

    source_names = ["pybullet"]

    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_grasp_center_joint"

    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")

    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"joint vel limits: {joint_vel_limits}")
    
    amplitudes = [0, 0.1, 0]
    frequencies = [0.4, 0.5, 0.4]

    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency,init_cartesian_pos)
    
    time_step = sim.GetTimeStep()
    current_time = 0
    cmd = MotorCommands()

    kp_pos = 100
    kp_ori = 100
    kp = 1000
    kd = 100

    for _ in range(5000):
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qd_des_clip = np.zeros_like(qd_mes)
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, init_joint_angles, qd_des_clip, kp, kd)
        sim.Step(cmd, "torque")

    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    
    regressor_all = np.array([])

    grasp_x, grasp_y, grasp_z, grasp_angle, grasp_width = sim.predict_grasp()
    draw_grasp_point(sim, grasp_x, grasp_y, grasp_z)
    p_d = np.array([grasp_x, grasp_y, grasp_z])
    pd_d = np.zeros_like(p_d)

    baseEuler = [0, -math.pi , 0]
    quat_values = sim.pybullet_client.getQuaternionFromEuler(baseEuler)
    ori_des = pin.Quaternion(quat_values[3], quat_values[0], quat_values[1], quat_values[2])
    
    while True:
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

        ori_d_des = None
        q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, p_d, pd_d, ori_des, ori_d_des, time_step, "both",  kp_pos, kp_ori, np.array(joint_vel_limits))
        print("inverse kinematic,",q_des[0])

        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
        sim.Step(cmd, "torque")

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des_clip)

        current_time += time_step
    
    num_joints = len(q_mes)
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    big_regressor = np.array(regressor_all)

if __name__ == '__main__':
    main()
