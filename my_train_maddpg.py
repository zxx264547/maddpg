from ieee123bus_env import IEEE123bus
from MADDPG import MADDPG
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.converter as pc

def create_123bus(pv_buses, es_buses):
    pp_net = pc.from_mpc('pandapower models/pandapower models/case_123.mat', casename_mpc_file='case_mpc')

    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    for bus in pv_buses:
        if bus in pp_net.bus.index:
            pp.create_sgen(pp_net, bus, p_mw=1.0, q_mvar=0.0)
        else:
            print(f"  Warning: Bus {bus} not found in network bus index")

    for bus in es_buses:
        if bus in pp_net.bus.index:
            pp.create_storage(pp_net, bus=bus, p_mw=0.5, max_e_mwh=2.0, soc_percent=50, min_e_mwh=0, q_mvar=0.1)
        else:
            print(f"  Warning: Bus {bus} not found in network bus index")

    return pp_net


# 定义 PV 和 ES 节点
pv_buses = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60])
es_buses = np.array([20, 30, 40])

# 创建 Pandapower 网络对象
pp_net = create_123bus(pv_buses, es_buses)

# 定义智能体参数
pv_params = (5, 1, 64)  # 观测维度，动作维度，隐藏层维度
storage_params = (5, 2, 64)  # 观测维度，动作维度，隐藏层维度

# 创建 MADDPG 实例
maddpg = MADDPG(pv_params, storage_params, pv_buses, es_buses, gamma=0.99, beta=0.001, tau=0.01, buffer_size=100000, batch_size=64)

# # 训练模型并记录电压数据
# voltage_data, over_limit_rates = maddpg.train(num_episodes=500, pp_net=pp_net, pv_bus=pv_buses, es_bus=es_buses)
# # 保存模型
# maddpg.save_model('model_directory')
#
# # 绘制电压越线情况
# voltage_data = np.array(voltage_data)
# num_steps = voltage_data.shape[0]
# num_buses = voltage_data.shape[1]

# 加载预训练模型
maddpg.load_model('model_directory')

# 在线训练模型并记录电压数据
voltage_data, over_limit_rates, alltime_voltage_values, alltime_pv_rewards, alltime_en_rewards, all_time_pv_actions, all_time_en_p_actions, all_time_en_q_actions = maddpg.train(num_episodes=1000, pp_net=pp_net, pv_bus=pv_buses, es_bus=es_buses)


# 绘制电压越限率随时间的变化
plt.figure(figsize=(10, 5))
plt.plot(over_limit_rates, label='Voltage Limit Exceedance Rate')
plt.xlabel('Step')
plt.ylabel('Over Limit Rate')
plt.title('Voltage Limit Exceedance Rate Over Time')
plt.legend()
plt.show()

alltime_voltage_values = np.array(alltime_voltage_values)
alltime_pv_rewards = np.array(alltime_pv_rewards)
alltime_en_rewards = np.array(alltime_en_rewards)
alltime_pv_actions = np.array(all_time_pv_actions)
alltime_en_p_actions = np.array(all_time_en_p_actions)
alltime_en_q_actions = np.array(all_time_en_q_actions)
# 选择要绘图的节点索引 (如第3个节点)
node_index = 2  # 假设我们选择第 3 个节点，Python 索引从 0 开始
# 提取该节点在所有 step 的电压变化
node_voltage_over_time = alltime_voltage_values[node_index, :]
# 生成 step 数组（横轴）
steps = range(alltime_voltage_values.shape[1])
# 绘制电压变化曲线
plt.figure(figsize=(10, 6))
plt.plot(steps, node_voltage_over_time, label=f'Node {node_index + 1} Voltage')
plt.xlabel('Step')
plt.ylabel('Voltage (p.u.)')
plt.title(f'Voltage Change for Node {node_index + 1} Over Time')
plt.legend()
plt.grid(True)
plt.show()

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