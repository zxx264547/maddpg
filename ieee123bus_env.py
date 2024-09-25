import gym
from gym import spaces
import numpy as np
import pandapower as pp

# 功能：
# 初始化
# 潮流计算
# 计算单个奖励
class IEEE123bus(gym.Env):
    def __init__(self, pp_net, pv_bus, es_bus, v0=1, vmax=1.05, vmin=0.95):
        super(IEEE123bus, self).__init__()

        self.network = pp_net  # pp_net 是通过 create_123bus() 函数创建的 Pandapower 网络对象。
        # self.obs_dim = 5  # 观测维度（输入维度）
        # self.pv_action_dim = 1  # 光伏动作维度（输出维度）
        # self.es_action_dim = 2  # 储能动作维度（输出维度）
        self.pv_buses = list(pv_bus)  # 光伏智能体控制的节点
        self.es_buses = list(es_bus)  # 储能智能体控制的节点
        # 存储光伏和储能节点的索引
        self.pv_buses_index = list(range(0, len(pv_bus)))
        self.es_buses_index = list(range(0, len(es_bus)))
        # 初始电压值以及电压阈值
        self.v0 = v0
        self.vmax = vmax
        self.vmin = vmin

        # # 定义光伏智能体的状态空间和动作空间
        # #TODO：这里的动作空间是归一化的值还是实际值好一点？
        # self.pv_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # self.pv_action_space = spaces.Box(low=-1, high=1, shape=(self.pv_action_dim,), dtype=np.float32)
        #
        # # 定义储能智能体的状态空间和动作空间
        # self.storage_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # self.storage_action_space = spaces.Box(low=-1, high=1, shape=(self.es_action_dim,), dtype=np.float32)

        # 初始化环境状态
        self.state_pv = []
        self.state_storage = []

    def reset_agent(self):
        # 基于潮流计算得到agent的初始局部状态值
        pp.runpp(self.network)
        self.state_pv = [self._get_state('pv', bus) for bus in self.pv_buses]
        self.state_storage = [self._get_state('storage', bus) for bus in self.es_buses]
        return self.state_pv, self.state_storage

    def step_agent(self,pv_actions, es_actions):
        pv_next_states = []
        pv_rewards = []
        pv_dones = []
        es_next_states = []
        es_rewards = []
        es_dones = []
        for i, bus in enumerate(self.pv_buses):
            self._apply_action(pv_actions[i], 'pv', bus)
        for i, bus in enumerate(self.es_buses):
            self._apply_action(es_actions[i], 'storage', bus)
        try:
            pp.runpp(self.network, max_iteration=500)
        except pp.powerflow.LoadflowNotConverged:
            print("Power flow for PV did not converge")
            # return -1, -99999, [True] * len(self.pv_buses), -1, -99999, [True] * len(self.es_buses),  # 终止当前回合

        # # 潮流计算成功，获得所有节点电压，用于计算单个智能体的reward
        # all_voltage_values = self.network.res_bus['vm_pu'].to_numpy()
        # 计算pv的reward和next_state
        for i, bus in enumerate(self.pv_buses):
            next_state = self._get_state('pv', bus)
            reward = self._calculate_reward(next_state)
            done = self._check_done(next_state)
            pv_next_states.append(next_state)
            pv_rewards.append(reward)
            pv_dones.append(done)
        # 计算es的reward和next_state
        for i, bus in enumerate(self.es_buses):
            next_state = self._get_state('storage', bus)
            reward = self._calculate_reward(next_state)
            done = self._check_done(next_state)
            es_next_states.append(next_state)
            es_rewards.append(reward)
            es_dones.append(done)
        # print(f"step agent success")
        return pv_next_states, pv_rewards, pv_dones, es_next_states, es_rewards, es_dones

    def _get_state(self, agent_type, bus):
        if agent_type == 'pv':
            if bus in self.network.sgen['bus'].values:
                sgen_idx = self.network.sgen[self.network.sgen['bus'] == bus].index[0]
                p_mw = self.network.sgen.at[sgen_idx, 'p_mw']
                q_mvar = self.network.sgen.at[sgen_idx, 'q_mvar']
                v_pu = self.network.res_bus.at[bus, 'vm_pu']
                load_p_mw = self.network.load.at[bus, 'p_mw'] if bus in self.network.load['bus'].values else 0.0
                load_q_mvar = self.network.load.at[bus, 'q_mvar'] if bus in self.network.load['bus'].values else 0.0

                # print(f" get local state: pv_id: {sgen_idx}, p_mw: {p_mw}, q_mvar:{q_mvar}, load_p_mw:{load_p_mw}, load_q_mvar:{load_q_mvar}, v_pu:{v_pu}")
                return np.array([p_mw, q_mvar, load_p_mw, load_q_mvar, v_pu])
            else:
                print(f"Warning: Bus {bus} not found in sgen")
                return None
        elif agent_type == 'storage':
            if bus in self.network.storage['bus'].values:
                storage_idx = self.network.storage[self.network.storage['bus'] == bus].index[0]
                p_mw = self.network.storage.at[storage_idx, 'p_mw']
                q_mvar = self.network.storage.at[storage_idx, 'q_mvar']
                v_pu = self.network.res_bus.at[bus, 'vm_pu']
                load_p_mw = self.network.load.at[bus, 'p_mw'] if bus in self.network.load['bus'].values else 0.0
                load_q_mvar = self.network.load.at[bus, 'q_mvar'] if bus in self.network.load['bus'].values else 0.0

                # print(f" get local state: en_id: {storage_idx}, p_mw: {p_mw}, q_mvar:{q_mvar}, load_p_mw:{load_p_mw}, load_q_mvar:{load_q_mvar}, v_pu:{v_pu}")
                return np.array([p_mw, q_mvar, load_p_mw, load_q_mvar, v_pu])
            else:
                print(f"Warning: Bus {bus} not found in storage")
                return None

    def _apply_action(self, action, agent_type, bus):
        if agent_type == 'pv':
            if bus in self.network.sgen['bus'].values:
                sgen_idx = self.network.sgen[self.network.sgen['bus'] == bus].index[0]
                # TODO:是否需要确保动作在合理范围内
                # q_mvar = np.clip(float(action), -1.0, 1.0)
                # self.network.sgen.loc[sgen_idx, 'q_mvar'] = q_mvar
                self.network.sgen.loc[sgen_idx, 'q_mvar'] = float(action)
            else:
                print(f"Warning: Bus {bus} not found in sgen")
        elif agent_type == 'storage':
            if bus in self.network.storage['bus'].values:
                storage_idx = self.network.storage[self.network.storage['bus'] == bus].index[0]
                # TODO:是否需要确保动作在合理范围内
                # p_mw = np.clip(float(action[0]), -1.0, 1.0)
                # q_mvar = np.clip(float(action[1]), -1.0, 1.0)
                # self.network.storage.loc[storage_idx, 'p_mw'] = p_mw
                # self.network.storage.loc[storage_idx, 'q_mvar'] = q_mvar
                self.network.storage.loc[storage_idx, 'p_mw'] = float(action[0])
                self.network.storage.loc[storage_idx, 'q_mvar'] = float(action[1])
            else:
                print(f"Warning: Bus {bus} not found in storage")

    def _calculate_reward(self, next_state):
        voltage = next_state[-1]
        # 对于低于下限的电压，计算为 lower_limit - voltage
        low_voltage_violations = np.maximum(self.vmin - voltage, 0)
        # 对于高于上限的电压，计算为 voltage - upper_limit
        high_voltage_violations = np.maximum(voltage - self.vmax, 0)
        # 计算总的电压越限程度
        voltage_violations = low_voltage_violations + high_voltage_violations
        reward = -voltage_violations * 1000
        # reward是一个标量
        return reward

    def _check_done(self, state):
        voltage = state[-1]
        return self.vmin < voltage < self.vmax

