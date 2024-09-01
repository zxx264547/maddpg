from numpy.ma.core import arange
from ieee123bus_env import IEEE123bus
from MADDPG import MADDPG
from DataVisualization import DataPlot
import numpy as np
import pandapower as pp
import pandapower.converter as pc

def create_123bus(pv_buses, es_buses):
    pp_net = pc.from_mpc('pandapower models/pandapower models/case_123.mat', casename_mpc_file='case_mpc')

    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    for bus in pv_buses:
        if bus in pp_net.bus.index:
            pp.create_sgen(pp_net, bus, p_mw=1.0, q_mvar=1.0)
        else:
            print(f"  Warning: Bus {bus} not found in network bus index")

    for bus in es_buses:
        if bus in pp_net.bus.index:
            pp.create_storage(pp_net, bus=bus, p_mw=0.5, max_e_mwh=2.0, soc_percent=50, min_e_mwh=0, q_mvar=0.1)
        else:
            print(f"  Warning: Bus {bus} not found in network bus index")
    print(f" create_123bus created")
    return pp_net


# 定义 PV 和 ES 节点
pv_buses = np.array([39, 47, 60]) - 1
es_buses = np.array([4, 85, 114]) - 1

# 创建 Pandapower 网络对象
pp_net = create_123bus(pv_buses, es_buses)

# 定义智能体参数
pv_params = (5, 1, 64)  # 观测维度，动作维度，隐藏层维度
storage_params = (5, 2, 64)  # 观测维度，动作维度，隐藏层维度

# 创建 MADDPG 实例
maddpg = MADDPG(pv_params, storage_params, pv_buses, es_buses, gamma=0.9, beta=0.9, tau=0.01, buffer_size=100000, batch_size=64)

# 训练模型并记录电压数据
result = maddpg.train(num_episodes=1000, pp_net=pp_net, pv_bus=pv_buses, es_bus=es_buses)
# 保存模型
maddpg.save_model('model_directory')


# # 加载预训练模型
# maddpg.load_model('model_directory')
#
# # 在线训练模型并记录电压数据
# # result = maddpg.train(num_episodes=1000, pp_net=pp_net, pv_bus=pv_buses, es_bus=es_buses)

# 解包元组
(voltage_data,
 voltage_violation_rates,
 alltime_pv_rewards,
 alltime_en_rewards,
 all_time_pv_actions,
 all_time_en_p_actions,
 all_time_en_q_actions) = result
# 数据可视化
DataPlot.rewards_step(alltime_pv_rewards, alltime_en_rewards)
DataPlot.pv_action_step(all_time_pv_actions)
DataPlot.en_p_action_step(all_time_en_p_actions)
DataPlot.en_q_action_step(all_time_en_q_actions)
DataPlot.voltage_step(voltage_data, arange(50))
DataPlot.voltage_violation_rates_step(voltage_violation_rates)
