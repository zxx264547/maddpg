import numpy as np
import matplotlib.pyplot as plt

class DataPlot:
    # voltage_data：每一行代表一个step，每一列代表一个节点
    # selected_nodes：从0开始
    @classmethod
    def voltage_step(cls, voltage_data, selected_nodes):
        # 将 Python 列表转换为 NumPy 数组
        voltage_data = np.array(voltage_data)
        # 获取时间步索引
        steps = range(voltage_data.shape[0])  # 假设时间步是从0到N-1
        # 绘制每个选定节点的电压变化曲线
        plt.figure(figsize=(10, 6))
        for node_index in selected_nodes:
            # 提取每个节点的电压数据
            node_voltage_over_time = voltage_data[:, node_index]
            plt.plot(steps, node_voltage_over_time, marker='o', linestyle='-', label=f'Node {node_index + 1} Voltage')
        # 图形标签和标题
        plt.xlabel('Step')  # 横坐标是时间步
        plt.ylabel('Voltage (p.u.)')  # 纵坐标是电压
        plt.title('Voltage Change for Selected Nodes Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    @classmethod
    def voltage_violation_rates_step(cls, voltage_violation_rates):
        # 绘制电压越限率随时间的变化
        plt.figure(figsize=(10, 5))
        plt.plot(voltage_violation_rates)
        plt.xlabel('Step')
        plt.ylabel('电压越限率')
        plt.title('电压越限率变化')
        plt.legend()
        plt.show()

    @classmethod
    def pv_action_step(cls, alltime_pv_actions):
        alltime_pv_actions = np.array(alltime_pv_actions)
        # 获取 step 和 PV 节点的数量
        num_steps = alltime_pv_actions.shape[0]
        num_pv_nodes = alltime_pv_actions.shape[1]
        # 生成 step 数组（横轴）
        steps = range(num_steps)
        # 绘制所有 PV 节点的动作变化
        plt.figure(figsize=(10, 6))
        # 遍历每个 PV 节点，绘制其动作变化曲线
        for pv_index in range(num_pv_nodes):
            pv_actions_over_time = alltime_pv_actions[:, pv_index]  # 提取第 pv_index 列的数据
            plt.plot(steps, pv_actions_over_time, label=f'PV Node {pv_index + 1}')
        # 添加图例、标题和标签
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.title('PV Node Actions Over Time')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()

    @classmethod
    def en_p_action_step(cls, alltime_en_p_actions):
        alltime_en_p_actions = np.array(alltime_en_p_actions)
        # 获取 step 和 en 节点的数量
        num_steps = alltime_en_p_actions.shape[0]
        num_en_nodes = alltime_en_p_actions.shape[1]
        # 生成 step 数组（横轴）
        steps = range(num_steps)
        # 绘制所有 PV 节点的动作变化
        plt.figure(figsize=(10, 6))
        # 遍历每个 PV 节点，绘制其动作变化曲线
        for en_index in range(num_en_nodes):
            en_p_actions_over_time = alltime_en_p_actions[:, en_index]  # 提取第 pv_index 列的数据
            plt.plot(steps, en_p_actions_over_time, label=f'PV Node {en_index + 1}')
        # 添加图例、标题和标签
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.title('en Node p Actions Over Time')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()

    @classmethod
    def en_q_action_step(cls, alltime_en_q_actions):
        alltime_en_q_actions = np.array(alltime_en_q_actions)
        # 获取 step 和 en 节点的数量
        num_steps = alltime_en_q_actions.shape[0]
        num_en_nodes = alltime_en_q_actions.shape[1]
        # 生成 step 数组（横轴）
        steps = range(num_steps)
        # 绘制所有 PV 节点的动作变化
        plt.figure(figsize=(10, 6))
        # 遍历每个 PV 节点，绘制其动作变化曲线
        for en_index in range(num_en_nodes):
            en_q_actions_over_time = alltime_en_q_actions[:, en_index]  # 提取第 pv_index 列的数据
            plt.plot(steps, en_q_actions_over_time, label=f'PV Node {en_index + 1}')
        # 添加图例、标题和标签
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.title('en Node q Actions Over Time')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()

    @classmethod
    def rewards_step(cls, alltime_pv_rewards, alltime_en_rewards):
        alltime_pv_rewards = np.array(alltime_pv_rewards)
        alltime_en_rewards = np.array(alltime_en_rewards)
        # 获取时间步的数量
        steps = range(alltime_pv_rewards.shape[0])  # 获取行数作为时间步
        # 创建图形
        plt.figure(figsize=(12, 6))
        # 遍历每个智能体（列）来绘制奖励值随时间步的变化
        for agent_index in range(alltime_pv_rewards.shape[1]):
            # 提取每个智能体的 PV 奖励值
            pv_rewards = alltime_pv_rewards[:, agent_index]
            # 绘制每个智能体的 PV 奖励曲线
            plt.plot(steps, pv_rewards, label=f'PV Agent {agent_index + 1} Rewards')

        for agent_index in range(alltime_en_rewards.shape[1]):
            # 提取每个智能体的 EN 奖励值
            en_rewards = alltime_en_rewards[:, agent_index]
            # 绘制每个智能体的 EN 奖励曲线
            plt.plot(steps, en_rewards, label=f'EN Agent {agent_index + 1} Rewards')
        # 图形标签和标题
        plt.xlabel('Step')
        plt.ylabel('Rewards')
        plt.title('PV and EN Rewards for Each Agent Over Time')
        plt.legend()
        plt.grid(True)
        # 显示图形
        plt.show()