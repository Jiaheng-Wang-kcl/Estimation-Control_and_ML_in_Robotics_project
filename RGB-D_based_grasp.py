import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
# import 
from utils.camera import Camera
from ggcnn.ggcnn import GGCNNNet, drawGrasps, drawRect, getGraspDepth
import skimage.transform as skt
import scipy.stats as ss


IMAGEWIDTH = 640
IMAGEHEIGHT = 480

# IMAGEWIDTH = 80
# IMAGEHEIGHT = 56

nearPlane = 0.01
farPlane = 10

fov = 60    # 垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT

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
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

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

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

class RGBD_Grasp_Env(pb.SimInterface):
    def __init__(self, conf_file_name: str, conf_file_path_ext: str = None):
        """
        初始化 RGBD_Grasp_Env 环境，继承自 SimInterface
        """
        super().__init__(conf_file_name, conf_file_path_ext)
        self.camera = Camera()  # 使用现有的Camera类进行相机参数的设置
        self.movecamera(0, 0, 2)

        self.projectionMatrix = self.pybullet_client.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def movecamera(self, x, y, z=0.7):
        """
        移动相机至指定位置，设置viewMatrix
        """
        self.camera_pos = np.array([x, y, z])
        self.viewMatrix = self.pybullet_client.computeViewMatrix(
            [x, y, z], [x, y, 0], [0, 1, 0]
        )
        self.camera_target = [x, y, 0]

    # def renderCameraDepthImage(self):
    #     """
    #     渲染计算抓取配置所需的深度图像
    #     """
    #     # 渲染图像
    #     t = time.time()
    #     # img_camera = self.pybullet_client.getCameraImage(
    #     #     IMAGEWIDTH, IMAGEHEIGHT, 
    #     #     self.viewMatrix, self.projectionMatrix, 
    #     #     # renderer=self.pybullet_client.ER_BULLET_HARDWARE_OPENGL
    #     #     renderer=self.pybullet_client.ER_TINY_RENDERER

    #     # )
    #     img_camera = self.pybullet_client.getCameraImage(
    #         IMAGEWIDTH, IMAGEHEIGHT,
    #         self.viewMatrix, self.projectionMatrix,
    #         shadow=False, renderer=self.pybullet_client.ER_TINY_RENDERER, flags=0
    #     )

    #     print("t1:",time.time() - t)
    #     t = time.time()
    #     w = img_camera[0]      # 图像宽度（像素）
    #     h = img_camera[1]      # 图像高度（像素）
    #     dep = img_camera[3]    # 深度数据


    #     # 获取深度图像
    #     depth = np.reshape(dep, (h, w))
    #     A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
    #     B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
    #     C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
    #     im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位：米
    #     print("t2:",time.time() - t)
    #     # t = time.time()
    #     return im_depthCamera
    
    def renderCameraDepthImage(self):
        """
        使用射线法渲染计算抓取配置所需的深度图像
        """
        t = time.time()
        camera_pos = self.camera_pos
        target_pos = self.camera_target

        # 计算相机视场的射线方向
        x = np.linspace(-1, 1, IMAGEWIDTH)
        y = np.linspace(-1, 1, IMAGEHEIGHT)
        xv, yv = np.meshgrid(x, y)
        directions = np.stack([xv, yv, np.ones_like(xv)], axis=-1).reshape(-1, 3)

        # 确保相机位置是 (3,) 形状的向量
        ray_from = [camera_pos] * len(directions)
        ray_to = [camera_pos + d * farPlane for d in directions]

        # 使用 PyBullet 的 rayTestBatch 批量计算射线交点
        results = self.pybullet_client.rayTestBatch(ray_from, ray_to)
        
        # 根据射线的结果计算深度图
        depth_map = np.array([r[2] if r[0] != -1 else farPlane for r in results]).reshape((IMAGEHEIGHT, IMAGEWIDTH))

        print("t2:", time.time() - t)
        return depth_map

    def gaussian_noise(self, im_depth):
        """
        在image上添加高斯噪声，参考dex-net代码
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = 0.002  # 0.002
        gaussian_process_scaling_factor = 8.0   # 8.0

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
        添加高斯噪声
        """
        img = self.gaussian_noise(img)
        return img

def main():
    # Configuration for the simulation
    conf_file_name = "panda_grasp_config.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface
    sim = RGBD_Grasp_Env(conf_file_name, conf_file_path_ext = cur_dir)
    camera_depth = sim.renderCameraDepthImage()
    camera_depth = sim.add_noise(camera_depth)
    # ggcnn = GGCNNNet('ggcnn/ckpt/epoch_0213_acc_0.6374.pth', device="cpu")    # 初始化ggcnn




    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    # print("Hi !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",init_joint_angles)
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    # print init joint
    print(f"Initial joint angles: {init_joint_angles}")
    
    # check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")


    joint_vel_limits = sim.GetBotJointsVelLimit()
    
    print(f"joint vel limits: {joint_vel_limits}")
    

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [0, 0.1, 0]  # Example amplitudes for 4 joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4]  # Example frequencies for 4 joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency,init_cartesian_pos)  # Initialize the reference
    
    #check = ref.check_sinusoidal_feasibility(sim)  # Check the feasibility of the reference trajectory
    #if not check:
    #    raise ValueError("Sinusoidal reference trajectory is not feasible. Please adjust the amplitude or frequency.")
    
    #simulation_time = sim.GetTimeSinceReset()
    time_step = sim.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    
    # P conttroller high level
    kp_pos = 100 # position 
    kp_ori = 0   # orientation
    
    # PD controller gains low level (feedbacklingain)
    kp = 1000
    kd = 100

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    
    # regressor all is a list of matrices
    regressor_all = np.array([])

    
    # data collection loop
    while True:
        # measure current state
        camera_depth = sim.renderCameraDepthImage()
        # camera_depth = sim.add_noise(camera_depth)
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        p_d, pd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # inverse differential kinematics
        ori_des = None
        ori_d_des = None
        q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, p_d, pd_d, ori_des, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)  # Zero torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)): # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des_clip)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        print("current time in seconds",current_time)

    
    num_joints = len(q_mes)
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    # training procedure
    
    # Convert lists of matrices to NumPy arrays for easier manipulation in computations
    
    big_regressor = np.array(regressor_all)
     
    
    

if __name__ == '__main__':
    main()