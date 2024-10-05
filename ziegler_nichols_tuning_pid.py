import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "panda_ZN_config.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")



# single joint tuning
#episode_duration is specified in seconds
# def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
#     # here we reset the simulator each time we start a new test
#     sim_.ResetPose()
    
#     # updating the kp value for the joint we want to tune
#     kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
#     kp_vec[joints_id] = kp

#     kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
#     # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
#     q_des = init_joint_angles.copy()
#     qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

#     q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
#     time_step = sim_.GetTimeStep()
#     current_time = 0
#     # Command and control loop
#     cmd = MotorCommands()  # Initialize command structure for motors


#     # Initialize data storage
#     q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    

#     steps = int(episode_duration/time_step)
#     # testing loop
#     for i in range(steps):
#         # measure current state
#         q_mes = sim_.GetMotorAngles(0)
#         qd_mes = sim_.GetMotorVelocities(0)
#         qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
#         # Compute sinusoidal reference trajectory
#         # Ensure q_init is within the range of the amplitude
        
#         # Control command
#         cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
#         sim_.Step(cmd, "torque")  # Simulation step with torque command

#         # Exit logic with 'q' key
#         keys = sim_.GetPyBulletClient().getKeyboardEvents()
#         qKey = ord('q')
#         if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
#             break
        
#         #simulation_time = sim.GetTimeSinceReset()

#         # Store data for plotting
#         q_mes_all.append(q_mes)
#         qd_mes_all.append(qd_mes)
#         q_d_all.append(q_des)
#         qd_d_all.append(qd_des)
#         #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
#         #regressor_all = np.vstack((regressor_all, cur_regressor))

#         #time.sleep(0.01)  # Slow down the loop for better visualization
#         # get real time
#         current_time += time_step
#         #print("current time in seconds",current_time)

    
#     # if plot:
#     #     plt.figure()
#     #     plt.plot(np.arange(steps)*time_step, q_mes_all, label="Measured Position")
#     #     plt.plot(np.arange(steps)*time_step, q_d_all, label="Desired Position")
#     #     plt.xlabel("Time (s)")
#     #     plt.ylabel("Joint Position (rad)")
#     #     plt.title(f"Joint {joints_id} Position Response")
#     #     plt.legend()
#     #     plt.grid(True)
#     #     plt.show()
    
    
#     return q_mes_all
     

def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    sim_.ResetPose()
    
    kp_vec = np.array([1000] * dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    # kd = np.array([2.496] * dyn_model.getNumberofActuatedJoints())
    kd = np.array([5] * dyn_model.getNumberofActuatedJoints())
    
    q_des = init_joint_angles.copy()
    qd_des = np.array([0] * dyn_model.getNumberofActuatedJoints())
    
    # 使用恒定的阶跃输入
    q_des[joints_id] += regulation_displacement

    time_step = sim_.GetTimeStep()
    current_time = 0

    cmd = MotorCommands()  # Initialize command structure for motors


    q_mes_all, q_d_all = [], []

    steps = int(episode_duration / time_step)
    
    for i in range(steps):
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)

        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)
        sim_.Step(cmd, "torque")

        q_mes_all.append(q_mes[joints_id])
        q_d_all.append(q_des[joints_id])

        current_time += time_step

    if plot:
        plt.figure()
        plt.plot(np.arange(steps) * time_step, q_mes_all, label="Measured Position")
        plt.plot(np.arange(steps) * time_step, q_d_all, label="Desired Position", linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Position (rad)")
        plt.title(f"Joint {joints_id} Position Response (Kp={kp})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return q_mes_all




def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return xf, power


# TODO Implement the table in thi function

if __name__ == '__main__':
    # 参数设置
    joint_id = 0  # 调试的关节ID
    regulation_displacement = 1.0  # 关节位移
    # init_gain = 10.24  # 初始Kp增益
    init_gain = 10.24  # 初始Kp增益
    gain_step = 1.0  # 每次Kp递增的步长
    max_gain = 10000  # 最大Kp
    test_duration = 20  # 测试时长，单位秒
    
    # 初始化仿真
    # conf_file_name = "pandaconfig.json"
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir,connection_mode=pybullet.DIRECT)
    # sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    current_kp = init_gain
    result_data = []  # 存储每次测试的结果
    
    while current_kp <= max_gain:
        print(f"Testing with Kp = {current_kp}")
        
        # 调用simulate_with_given_pid_values测试当前的Kp值
        joint_angles = simulate_with_given_pid_values(sim, current_kp, joint_id, regulation_displacement, test_duration, plot=True)
        
        # 计算采样时间
        dt = sim.GetTimeStep()
        
        # 频率分析
        freqs, power = perform_frequency_analysis(joint_angles, dt)
        
        # 检查系统是否振荡
        is_oscillating = np.max(power) > 0.4  # 设定某个阈值判断是否振荡
        
        # # 保存结果
        # result_data.append({
        #     'Kp': current_kp,
        #     'frequency': freqs[np.argmax(power)],  # 振荡的频率
        #     'is_oscillating': is_oscillating
        # })
        
        # 如果系统已经开始振荡，停止Kp的递增
        if is_oscillating:
            print(f"System oscillating at Kp = {current_kp}")
            # break
        
        # 递增Kp
        current_kp *= gain_step
    


    # 输出结果
    for result in result_data:
        print(f"Kp: {result['Kp']}, Frequency: {result['frequency']}, Oscillating: {result['is_oscillating']}")
    
    # # 如果找到系统振荡的Kp，使用Ziegler-Nichols方法计算最终的PID参数
    # if result_data[-1]['is_oscillating']:
    #     Ku = result_data[-1]['Kp']  # 临界增益
    #     Pu = 1 / result_data[-1]['frequency']  # 振荡周期
    #     Kp_final = 0.6 * Ku
    #     Ki_final = 1.2 * Ku / Pu
    #     Kd_final = 0.075 * Ku * Pu
    #     print(f"Final PID parameters: Kp={Kp_final}, Ki={Ki_final}, Kd={Kd_final}")



# if __name__ == '__main__':
#     joint_id = 0  # Joint ID to tune
#     regulation_displacement = 1.0  # Displacement from the initial joint position
#     init_gain=1000 
#     gain_step=1.5 
#     max_gain=10000 
#     test_duration=20 # in seconds
    
#     conf_file_name = "pandaconfig.json"
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
#     # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
#     # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
#     simulate_with_given_pid_values(sim, 100, )
   