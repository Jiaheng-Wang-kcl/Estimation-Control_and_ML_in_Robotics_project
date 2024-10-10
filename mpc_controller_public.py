import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from regulator_model import RegulatorModel

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    

def getSystemMatrices(sim, num_joints, damping_coefficients=None):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    time_step = sim.GetTimeStep()
    
    # TODO: Finish the system matrices 
    # Initialize the state transition matrix A and control input matrix B
    A = np.eye(num_states)
    B = np.zeros((num_states, num_controls))
    
    # If damping coefficients are provided, include damping in the model
    if damping_coefficients is not None:
        # Assuming damping_coefficients is a 1D array or list with length equal to num_joints
        damping_matrix = np.diag(damping_coefficients)
        
        # Construct the A matrix with damping
        A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix
        A[:num_joints, num_joints:] = time_step * np.eye(num_joints)
        
        # Construct the B matrix
        B[num_joints:, :] = time_step * np.eye(num_joints)
    
    else:
        # No damping case
        A[:num_joints, num_joints:] = time_step * np.eye(num_joints)
        B[num_joints:, :] = time_step * np.eye(num_joints)
    # print("check A:\n",np.array2string(A, separator=', ', max_line_width=1000),"check B:\n",np.array2string(B, separator=', ', max_line_width=1000))
    # exit()
    
    return A, B


def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    # Q = 1 * np.eye(num_states)  # State cost matrix
    # Q = 1000 * np.eye(num_states)
    # Q = 10000 * np.eye(num_states)

    # Q = np.eye(num_states)
    # Q[:num_joints, :num_joints] = np.diag([105.44138259, 763.22633178, 170.52358618, 892.09899107, 490.70289289, 996.56302905, 525.60617237])  # 位置状态部分的对角线元素为遗传算法优化的值
    # Q[num_joints:, num_joints:] = np.diag([0.1] * num_joints)  # 仅速度状态部分的对角线设置为 0.1

    # Q = np.diag([5933.51276877, 8267.91068812, 2487.44018102, 4820.30739794, 5315.67675927, 5243.55716766, 2686.90707872,16.51727031, 36.62807419, 11.8795082, 
    #                  28.09746468, 44.79688295, 25.99829336, 54.84492368])
    Q = np.diag([11175.82850625, 24773.78924499,  9437.16661227, 21108.08381245, 9819.36754549, 24242.77122351 , 1215.0550485,
                 1577.15456977 , 168.76811945, 1525.41731318,  124.97577197, 1665.33206897,   33.13889389,  528.45077096])
    # Q[num_joints:, num_joints:] = 0.1
    
    # R = 0.1 * np.eye(num_controls)  # Control input cost matrix
    R = np.diag([18.48632463,  2.20182626,  3.97916945,  3.96988714, 12.24230534,  0.08303234,  3.33291095])
    # R = 20 * np.eye(num_controls)  # Control input cost matrix
    
    return Q, R


def main():
    # Configuration
    # conf_file_name = "pandaconfig.json" 
    conf_file_name = "panda_ZN_config.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    # Define the matrices
    # A, B = getSystemMatrices(sim, num_joints)
    # damping_coefficients = 1000 * np.ones(num_joints)
    damping_coefficients = None
    A, B = getSystemMatrices(sim, num_joints, damping_coefficients=damping_coefficients)
    Q, R = getCostMatrices(num_joints)
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # Main control loop
    # episode_duration = 5
    episode_duration = 20
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_joints]
       
        # Control command
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # print(cmd.tau_cmd)
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        print(f"Current time: {current_time}")
        # print(f"Current time: {current_time}, 1st joint: {q_mes[1]}")
    
    
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        ymin, ymax = plt.gca().get_ylim()
        ymin = min(ymin, -0.1)
        ymax = max(ymax, 0.1)
        plt.ylim(ymin, ymax)

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        ymin, ymax = plt.gca().get_ylim()
        ymin = min(ymin, -0.1)
        ymax = max(ymax, 0.1)
        plt.ylim(ymin, ymax)

        plt.tight_layout()
        plt.show()
    
     
    
    
if __name__ == '__main__':
    
    main()