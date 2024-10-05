# import numpy as np
# import time
# import os
# import matplotlib.pyplot as plt
# from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
# import pinocchio as pin

# def computeRegressor(pin_wrapper_instance, q_mes, qd_mes, qdd_mes):
#     """
#     计算动力学回归矩阵 (外部函数)
#     :param pin_wrapper_instance: PinWrapper 类的实例
#     :param q_mes: 当前的关节位置
#     :param qd_mes: 当前的关节速度
#     :param qdd_mes: 当前的关节加速度
#     :return: 动力学回归矩阵
#     """
#     # 调用 ReoderJoints2PinVec 来重排关节状态以匹配 Pinocchio 模型
#     q_ = pin_wrapper_instance.ReoderJoints2PinVec(q_mes, "pos")
#     qd_ = pin_wrapper_instance.ReoderJoints2PinVec(qd_mes, "vel")
#     qdd_ = pin_wrapper_instance.ReoderJoints2PinVec(qdd_mes, "vel")
    
#     # 使用 Pinocchio 的 computeJointTorqueRegressor 计算回归矩阵
#     regressor = pin.computeJointTorqueRegressor(pin_wrapper_instance.pin_model, pin_wrapper_instance.pin_data, q_, qd_, qdd_)
    
#     return regressor

# def main():
#     # Configuration for the simulation
#     conf_file_name = "pandaconfig.json"  # Configuration file for the robot
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

#     # Get active joint names from the simulation
#     ext_names = sim.getNameActiveJoints()
#     ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

#     source_names = ["pybullet"]  # Define the source for dynamic modeling

#     # Create a dynamic model of the robot
#     dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
#     num_joints = dyn_model.getNumberofActuatedJoints()

#     # Print initial joint angles
#     print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

#     # Sinusoidal reference
#     amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
#     frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints
#     amplitude = np.array(amplitudes)
#     frequency = np.array(frequencies)
#     ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference

#     # Simulation parameters
#     time_step = sim.GetTimeStep()
#     current_time = 0
#     max_time = 10  # seconds

#     # Command and control loop
#     cmd = MotorCommands()  # Initialize command structure for motors
#     # PD controller gains
#     kp = 1000
#     kd = 100

#     # Initialize data storage
#     tau_mes_all = []
#     regressor_all = []

#     # Data collection loop
#     while current_time < max_time:
#         # Measure current state
#         q_mes = sim.GetMotorAngles(0)
#         qd_mes = sim.GetMotorVelocities(0)
#         qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

#         # Compute sinusoidal reference trajectory
#         q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity

#         # Control command
#         cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
#         sim.Step(cmd, "torque")

#         # Get measured torque
#         tau_mes = sim.GetMotorTorques(0)

#         if dyn_model.visualizer:
#             for index in range(len(sim.bot)):  # Conditionally display the robot model
#                 q = sim.GetMotorAngles(index)
#                 dyn_model.DisplayModel(q)  # Update the display of the robot model

#         # Exit logic with 'q' key
#         keys = sim.GetPyBulletClient().getKeyboardEvents()
#         qKey = ord('q')
#         if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
#             break

#         # Compute regressor and store it
#         regressor = computeRegressor(dyn_model, q_mes, qd_mes, qdd_mes)
#         regressor_all.append(regressor)
#         tau_mes_all.append(tau_mes)

#         current_time += time_step
#         print(f"Current time in seconds: {current_time:.2f}")

#     # Stack all the regressors and torques
#     regressor_all = np.vstack(regressor_all)  # Stack all regressors
#     tau_mes_all = np.vstack(tau_mes_all)  # 将所有时间步的扭矩值堆叠为二维数组，形状为 (time_steps, num_joints)

#     # Compute the parameters 'theta' using pseudoinverse
#     theta = np.linalg.pinv(regressor_all).dot(tau_mes_all.flatten())  # theta 计算需要展平后的扭矩值
#     print(f"Estimated parameters: {theta}")

#     # Compute the torque prediction
#     tau_pred = regressor_all.dot(theta).reshape(tau_mes_all.shape)  # 将预测值还原为二维数组，形状为 (time_steps, num_joints)

#     # Calculate the error for each joint and compute mean squared error (MSE)
#     error = tau_mes_all - tau_pred
#     mse_per_joint = np.mean(error**2, axis=0)  # 计算每个关节的均方误差
#     total_mse = np.mean(mse_per_joint)  # 总均方误差
#     print(f"Mean squared error of the model (per joint): {mse_per_joint}")
#     print(f"Total mean squared error of the model: {total_mse}")

#     # Plot torque prediction error for each joint
#     plt.figure()
#     for i in range(num_joints):
#         plt.subplot(num_joints, 1, i+1)
#         plt.plot(range(tau_mes_all.shape[0]), tau_mes_all[:, i], label='Measured torque')
#         plt.plot(range(tau_pred.shape[0]), tau_pred[:, i], label='Predicted torque')
#         plt.title(f'Joint {i+1} torque comparison')
#         plt.xlabel('Time step')
#         plt.ylabel('Torque')
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     main()






# import numpy as np
# import time
# import os
# import matplotlib.pyplot as plt
# from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
# import pinocchio as pin

# def computeRegressor(pin_wrapper_instance, q_mes, qd_mes, qdd_mes):
#     """
#     计算动力学回归矩阵 (外部函数)
#     :param pin_wrapper_instance: PinWrapper 类的实例
#     :param q_mes: 当前的关节位置
#     :param qd_mes: 当前的关节速度
#     :param qdd_mes: 当前的关节加速度
#     :return: 动力学回归矩阵
#     """
#     q_ = pin_wrapper_instance.ReoderJoints2PinVec(q_mes, "pos")
#     qd_ = pin_wrapper_instance.ReoderJoints2PinVec(qd_mes, "vel")
#     qdd_ = pin_wrapper_instance.ReoderJoints2PinVec(qdd_mes, "vel")
    
#     regressor = pin.computeJointTorqueRegressor(pin_wrapper_instance.pin_model, pin_wrapper_instance.pin_data, q_, qd_, qdd_)
#     return regressor

# def main():
#     # Configuration for the simulation
#     conf_file_name = "pandaconfig.json"
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)

#     ext_names = sim.getNameActiveJoints()
#     ext_names = np.expand_dims(np.array(ext_names), axis=0)

#     source_names = ["pybullet"]
#     dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
#     num_joints = dyn_model.getNumberofActuatedJoints()

#     print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

#     amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]
#     frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
#     amplitude = np.array(amplitudes)
#     frequency = np.array(frequencies)
#     ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())

#     time_step = sim.GetTimeStep()
#     current_time = 0
#     max_time = 10

#     cmd = MotorCommands()
#     kp = 1000
#     kd = 100

#     tau_mes_all = []
#     regressor_all = []

#     while current_time < max_time:
#         q_mes = sim.GetMotorAngles(0)
#         qd_mes = sim.GetMotorVelocities(0)
#         qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

#         q_d, qd_d = ref.get_values(current_time)

#         cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
#         sim.Step(cmd, "torque")

#         tau_mes = sim.GetMotorTorques(0)

#         if dyn_model.visualizer:
#             for index in range(len(sim.bot)):
#                 q = sim.GetMotorAngles(index)
#                 dyn_model.DisplayModel(q)

#         keys = sim.GetPyBulletClient().getKeyboardEvents()
#         qKey = ord('q')
#         if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
#             break

#         regressor = computeRegressor(dyn_model, q_mes, qd_mes, qdd_mes)
#         regressor_all.append(regressor)
#         tau_mes_all.append(tau_mes)

#         current_time += time_step
#         print(f"Current time in seconds: {current_time:.2f}")

#     regressor_all = np.vstack(regressor_all)
#     tau_mes_all = np.vstack(tau_mes_all)

#     # 计算参数 'theta'
#     theta = np.linalg.pinv(regressor_all).dot(tau_mes_all.flatten())
#     print(f"Estimated parameters: {theta}")

#     # 使用更新的参数重新计算扭矩预测
#     tau_pred = regressor_all.dot(theta).reshape(tau_mes_all.shape)

#     # 计算误差
#     error = tau_mes_all - tau_pred
#     mse_per_joint = np.mean(error**2, axis=0)
#     total_mse = np.mean(mse_per_joint)
#     print(f"Mean squared error of the model (per joint): {mse_per_joint}")
#     print(f"Total mean squared error of the model: {total_mse}")

#     # 重新用更新后的参数重新计算所有数据并计算总误差
#     print("\nRe-calculating torque predictions using the estimated parameters...")

#     # 使用相同的回归矩阵和估计的 theta 重新计算预测扭矩
#     tau_pred_recalculated = regressor_all.dot(theta).reshape(tau_mes_all.shape)

#     # 计算重新计算的误差
#     error_recalculated = tau_mes_all - tau_pred_recalculated
#     mse_per_joint_recalculated = np.mean(error_recalculated**2, axis=0)
#     total_mse_recalculated = np.mean(mse_per_joint_recalculated)

#     print(f"Recalculated mean squared error of the model (per joint): {mse_per_joint_recalculated}")
#     print(f"Recalculated total mean squared error of the model: {total_mse_recalculated}")

#     # 绘制扭矩预测与实际值的对比
#     plt.figure()
#     for i in range(num_joints):
#         plt.subplot(num_joints, 1, i+1)
#         plt.plot(range(tau_mes_all.shape[0]), tau_mes_all[:, i], label='Measured torque')
#         plt.plot(range(tau_pred_recalculated.shape[0]), tau_pred_recalculated[:, i], label='Predicted torque')
#         plt.title(f'Joint {i+1} torque comparison')
#         plt.xlabel('Time step')
#         plt.ylabel('Torque')
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     main()

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
import pinocchio as pin

def computeRegressor(pin_wrapper_instance, q_mes, qd_mes, qdd_mes):
    q_ = pin_wrapper_instance.ReoderJoints2PinVec(q_mes, "pos")
    qd_ = pin_wrapper_instance.ReoderJoints2PinVec(qd_mes, "vel")
    qdd_ = pin_wrapper_instance.ReoderJoints2PinVec(qdd_mes, "vel")
    regressor = pin.computeJointTorqueRegressor(pin_wrapper_instance.pin_model, pin_wrapper_instance.pin_data, q_, qd_, qdd_)
    return regressor

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)

    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)

    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())

    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10

    cmd = MotorCommands()
    kp = 1000
    kd = 100

    tau_mes_all = []
    regressor_all = []

    while current_time < max_time:
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        q_d, qd_d = ref.get_values(current_time)

        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        if current_time > 1.0:  
            regressor = computeRegressor(dyn_model, q_mes, qd_mes, qdd_mes)
            regressor_all.append(regressor)
            tau_mes_all.append(tau_mes)

        current_time += time_step
        print(f"Current time in seconds: {current_time:.2f}")

    regressor_all = np.vstack(regressor_all)
    print("shape of regressor_all",regressor_all.shape)
    tau_mes_all = np.vstack(tau_mes_all)
    print("shape of tau_mes_all",tau_mes_all.shape)

    theta = np.linalg.pinv(regressor_all).dot(tau_mes_all.flatten())
    print(f"Estimated parameters: {theta}")

    tau_pred = regressor_all.dot(theta).reshape(tau_mes_all.shape)

    error = tau_mes_all - tau_pred
    mse_per_joint = np.mean(error**2, axis=0)
    total_mse = np.mean(mse_per_joint)
    print(f"Mean squared error of the model (per joint): {mse_per_joint}")
    print(f"Total mean squared error of the model: {total_mse}")

    plt.figure()
    for i in range(num_joints):
        plt.subplot(num_joints, 1, i+1)
        plt.plot(range(tau_mes_all.shape[0]), tau_mes_all[:, i], label='Measured torque')
        plt.plot(range(tau_pred.shape[0]), tau_pred[:, i], label='Predicted torque')
        plt.title(f'Joint {i+1} torque comparison')
        plt.xlabel('Time step')
        plt.ylabel('Torque')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
