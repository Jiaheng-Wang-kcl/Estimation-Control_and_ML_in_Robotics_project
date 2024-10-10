import os
import numpy as np
import random
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from regulator_model import RegulatorModel
from mpc_controller_public import getSystemMatrices

# 设置遗传算法参数
population_size = 50
num_generations = 30
initial_mutation_rate = 0.3  # 初始较高的变异率
min_mutation_rate = 0.1      # 最低变异率
num_joints = 7  # 假设有 7 个关节

# 初始化仿真
def initialize_simulation(conf_file_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    return sim, dyn_model

# 评估个体适应度，返回关节角度的平方和总和
def evaluate_fitness(Q_diag_pos, Q_diag_vel, R_diag):
    sim, dyn_model = initialize_simulation("panda_ZN_config.json")
    cmd = MotorCommands()
    
    # 定义 A 和 B 矩阵
    damping_coefficients = None
    A, B = getSystemMatrices(sim, num_joints, damping_coefficients=damping_coefficients)
    
    # 定义 Q 矩阵为对角矩阵，分别设置位置和速度的调整范围
    Q = np.eye(2 * num_joints)
    Q[:num_joints, :num_joints] = np.diag(Q_diag_pos)
    Q[num_joints:, num_joints:] = np.diag(Q_diag_vel)
    
    # 定义 R 矩阵
    R = np.diag(R_diag)

    # 初始化 RegulatorModel
    num_states = 2 * num_joints
    C = np.eye(num_states)
    N_mpc = 10
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # 仿真主循环，记录关节角度的平方和
    episode_duration = 20  # 20秒仿真时间
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()
    
    squared_sum = 0
    for _ in range(steps):
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        x0_mpc = np.hstack((q_mes, qd_mes))
        
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[:num_joints]
        
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")
        
        # 仅累加位置的平方和
        squared_sum += np.sum(np.square(q_mes))
    
    return squared_sum

# 初始化种群（位置、速度和 R 矩阵部分）
def initialize_population():
    Q_pos_population = [np.random.uniform(0, 25000, num_joints) for _ in range(population_size)]
    Q_vel_population = [np.random.uniform(0, 2000, num_joints) for _ in range(population_size)]
    R_population = [np.random.uniform(0, 20, num_joints) for _ in range(population_size)]
    return list(zip(Q_pos_population, Q_vel_population, R_population))

# 选择函数
def select_parents(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)
    return [population[sorted_indices[0]], population[sorted_indices[1]]]

# 交叉和变异函数
def crossover(parent1, parent2):
    alpha = random.random()
    child_pos = alpha * parent1[0] + (1 - alpha) * parent2[0]
    child_vel = alpha * parent1[1] + (1 - alpha) * parent2[1]
    child_R = alpha * parent1[2] + (1 - alpha) * parent2[2]
    return (child_pos, child_vel, child_R)

def mutate(child, generation):
    # 动态衰减变异率
    mutation_rate = max(min_mutation_rate, initial_mutation_rate * (1 - generation / num_generations))
    
    if random.random() < mutation_rate:
        # 位置部分变异
        mutation_index_pos = random.randint(0, num_joints - 1)
        child[0][mutation_index_pos] += np.random.uniform(-0.5, 0.5) * child[0][mutation_index_pos]
        child[0][mutation_index_pos] = np.clip(child[0][mutation_index_pos], 0, 25000)
        
        # 速度部分变异
        mutation_index_vel = random.randint(0, num_joints - 1)
        child[1][mutation_index_vel] += np.random.uniform(-0.5, 0.5) * child[1][mutation_index_vel]
        child[1][mutation_index_vel] = np.clip(child[1][mutation_index_vel], 0, 2000)
        
        # R 矩阵部分变异
        mutation_index_R = random.randint(0, num_joints - 1)
        child[2][mutation_index_R] += np.random.uniform(-0.5, 0.5) * child[2][mutation_index_R]
        child[2][mutation_index_R] = np.clip(child[2][mutation_index_R], 0, 20)
    return child

# 主遗传算法循环
def genetic_algorithm():
    population = initialize_population()
    for generation in range(num_generations):
        fitness_scores = [evaluate_fitness(ind[0], ind[1], ind[2]) for ind in population]
        best_fitness = min(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        # 获取当前代中最佳个体
        best_individual = population[np.argmin(fitness_scores)]
        
        # 打印每代信息
        print(f"Generation {generation}, Best Fitness: {best_fitness}, Average Fitness: {avg_fitness}, Std Dev: {std_fitness}")
        print("Best Q-diagonal (position):", best_individual[0])
        print("Best Q-diagonal (velocity):", best_individual[1])
        print("Best R-diagonal:", best_individual[2])
        
        new_population = []
        parent1, parent2 = select_parents(population, fitness_scores)
        for _ in range(population_size):
            child = crossover(parent1, parent2)
            child = mutate(child, generation)
            new_population.append(child)
        
        population = new_population

    best_individual = population[np.argmin(fitness_scores)]
    print("Optimal Q-diagonal (position):", best_individual[0])
    print("Optimal Q-diagonal (velocity):", best_individual[1])
    print("Optimal R-diagonal:", best_individual[2])
    return best_individual

if __name__ == '__main__':
    genetic_algorithm()
