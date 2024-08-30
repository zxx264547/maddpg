from ieee123bus_env import IEEE123bus
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
"""
ReplayBuffer
PolicyNetwork
ValueNetwork
PVAgent
StorageAgent
MADDPG
"""


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        # 确保所有输入数据都是numpy数组，并且形状一致
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        # 打印每个数据的形状以验证一致性
        # print(f"Add: State Shape: {state.shape}, Action Shape: {action.shape}, Reward Shape: {reward.shape}, Next State Shape: {next_state.shape}, Done Shape: {done.shape}")

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #batch_size为一次抽取的样本数
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    #输入：输入维度，输出维度，隐藏层维度
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        #全连接层：输入-> 隐藏-> 隐藏-> 输出
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        #初始化权重和偏置项
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
        self.linear3.bias.data.uniform_(-3e-3, 3e-3)
    #forward是前向传播的过程，即给定输入获得输出
    #这里输入为state，即为观测值，输出为x，即为动作
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
        self.linear3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PVAgent:
    def __init__(self, id, obs_dim, action_dim, hidden_dim):
        self.id = id #光伏ID
        #初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.target_policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(obs_dim, action_dim, hidden_dim)
        self.target_value_net = ValueNetwork(obs_dim, action_dim, hidden_dim)
        #定义神经网络的优化器为Adam，输入参数为：神经网络的所有参数（权重和偏置），学习率（0.0001）
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=2e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        #定义损失函数，用于网络的优化
        self.value_criterion = nn.MSELoss()
        #检查是否有可用的GPU，并设置设备
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #将所有网络移动到计算设备（CPU或者GPU）
        self.policy_net.to(self.device)
        self.target_policy_net.to(self.device)
        self.value_net.to(self.device)
        self.target_value_net.to(self.device)

    #使用epsilon-贪心策略获取光伏的动作
    def get_action(self, state, epsilon=0.1):
        #以概率epsilon选择一个随机动作（探索）
        if random.random() < epsilon:
            # 生成一个在[-1,1]范围内的随机动作（动作维度为输出层的维度），概率为epsilon
            return np.random.uniform(-1, 1, self.policy_net.linear3.out_features)
        else:
            # 使用策略网络选择动作，概率为（1-epsilon）
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy_net.forward(state)
            return action.detach().cpu().numpy()[0]

    #目标网络的参数以软更新的方式更新
    def update_target_networks(self, soft_tau):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            # 软更新公式: target_param = target_param * (1 - soft_tau) + param * soft_tau
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)


class StorageAgent:
    def __init__(self, id, obs_dim, action_dim, hidden_dim):
        self.id = id

        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.target_policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(obs_dim, action_dim, hidden_dim)
        self.target_value_net = ValueNetwork(obs_dim, action_dim, hidden_dim
                                             )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=2e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.value_criterion = nn.MSELoss()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.policy_net.to(self.device)
        self.target_policy_net.to(self.device)
        self.value_net.to(self.device)
        self.target_value_net.to(self.device)

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return np.random.uniform(-1, 1, self.policy_net.linear3.out_features)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy_net.forward(state)
            return action.detach().cpu().numpy()[0]

    def update_target_networks(self, soft_tau):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)


class MADDPG:
    #参数：光伏参数（输入维度，隐藏维度，输出维度），储能参数，光伏节点，储能节点，折扣因子，目标网络的软更新参数，缓冲区大小，采样大小
    def __init__(self, pv_params, storage_params, pv_bus, es_bus, gamma, beta, tau, buffer_size, batch_size):
        # 计算总体reward相关变量
        # global_reward = total_power_loss + self_reward + beta * (sum_reward - self_reward)
        self.sum_reward = None
        self.total_power_loss = None
        self.beta = beta
        # end
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        #构造多智能体
        self.pv_agents = [PVAgent(i, *pv_params) for i in range(len(pv_bus))]
        self.storage_agents = [StorageAgent(i, *storage_params) for i in range(len(es_bus))]
        #初始化缓冲区
        self.pv_replay_buffer = ReplayBuffer(buffer_size)
        self.storage_replay_buffer = ReplayBuffer(buffer_size)

    # def update(self):
    #     if len(self.pv_replay_buffer) >= self.batch_size:
    #         states, actions, rewards, next_states, dones = self.pv_replay_buffer.sample(self.batch_size)
    #         self._update_agents(self.pv_agents, states, actions, rewards, next_states, dones)
    #
    #     if len(self.storage_replay_buffer) >= self.batch_size:
    #         states, actions, rewards, next_states, dones = self.storage_replay_buffer.sample(self.batch_size)
    #         self._update_agents(self.storage_agents, states, actions, rewards, next_states, dones)

    #智能体共享缓冲区
    #更新网络参数
    def update(self):
        self._update_agents(self.pv_agents, self.pv_replay_buffer)
        self._update_agents(self.storage_agents, self.storage_replay_buffer)

    def _update_agents(self, agents, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return

        batch = replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        for agent in agents:
            self._update_agent(agent, states, actions, rewards, next_states, dones)

    def _update_agent(self, agent, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(agent.device)
        actions = torch.FloatTensor(actions).to(agent.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(agent.device)
        next_states = torch.FloatTensor(next_states).to(agent.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(agent.device)
        #计算策略网络的损失
        policy_loss = self._compute_policy_loss(agent, states)
        #计算价值网络的损失
        value_loss = self._compute_value_loss(agent, states, actions, rewards, next_states, dones)

        #TODO:在论文中，把这里的更新步长变为N步
        #更新在线网络的参数（根据损失函数更新梯度）
        agent.policy_optimizer.zero_grad()
        policy_loss.backward()
        agent.policy_optimizer.step()

        agent.value_optimizer.zero_grad()
        value_loss.backward()
        agent.value_optimizer.step()
        #更新目标网络的参数
        agent.update_target_networks(self.tau)

    #根据当前的状态生成的动作来计算损失
    def _compute_policy_loss(self, agent, states):
        #通过观测值（状态）计算策略网络的生成动作
        policy_actions = agent.policy_net(states)
        #计算损失（通过最小化损失来更新参数）
        policy_loss = -agent.value_net(states, policy_actions).mean()
        return policy_loss
    #根据与目标网络的差异来更新在线价值网络
    def _compute_value_loss(self, agent, states, actions, rewards, next_states, dones):
        # 计算目标Q值
        with torch.no_grad():
            next_actions = agent.target_policy_net(next_states)
            target_q = agent.target_value_net(next_states, next_actions)
            #done为1时，表示终止状态，目标Q值为即时奖励；否则，要考虑未来奖励
            # TODO:这里的rewards要考虑总体的rewards（论文中rewards的改变也是在这里体现）
            global_reward = self.total_power_loss + self.beta * (self.sum_reward - rewards) + rewards
            target_q = global_reward + (1 - dones) * self.gamma * target_q

        # 计算当前Q值
        #TODO:此时的action应该是在线actor网络根据当前状态计算出来的，还是从采样池里采样出来的(目前的方案里面是采样池中的)
        current_q = agent.value_net(states, actions)

        # 计算价值网络的损失
        value_loss = agent.value_criterion(current_q, target_q)

        return value_loss

    def train(self, num_episodes, pp_net, pv_bus, es_bus):
        #创建配电网环境
        env = IEEE123bus(pp_net, pv_bus, es_bus)
        #初始化节点电压和越限率
        voltage_data = []
        over_limit_rates = []

        # 初始化缓冲区
        self._initialize_replay_buffer(env)

        for episode in range(num_episodes):
            #初始化环境（每个智能体的状态和总奖励）
            state_pv = env.reset_pv()
            state_storage = env.reset_storage()
            # 初始化越限率
            over_limit_rate = 1

            while over_limit_rate > 0.05:
                #根据当前状态获得每个智能体的动作
                actions_pv = [agent.get_action(state_pv[i]) for i, agent in enumerate(self.pv_agents)]
                actions_storage = [agent.get_action(state_storage[i]) for i, agent in enumerate(self.storage_agents)]
                #根据动作获得每个智能体的下一个状态和奖励（潮流计算）
                #TODO: 这里的先后顺序是否有影响？
                next_state_pv, rewards_pv, done_pv = env.step_pv(actions_pv)
                next_state_storage, rewards_storage, done_storage = env.step_storage(actions_storage)
                #将数据放入缓冲区
                for i, agent in enumerate(self.pv_agents):
                    self.pv_replay_buffer.add(state_pv[i], actions_pv[i], rewards_pv[i], next_state_pv[i], done_pv[i])
                for i, agent in enumerate(self.storage_agents):
                    self.storage_replay_buffer.add(state_storage[i], actions_storage[i], rewards_storage[i],
                                                   next_state_storage[i], done_storage[i])
                #更新状态
                state_pv = next_state_pv
                state_storage = next_state_storage

                self.sum_reward = sum(rewards_pv) + sum(rewards_storage)
                # 所有线路的有功损耗 + 无功损耗
                self.total_power_loss = pp_net.res_line.pl_mw.sum() + pp_net.res_line.ql_mvar.sum()
                # 记录电压数据并计算电压越限率
                voltage = env.network.res_bus.vm_pu.to_numpy()
                voltage_data.append(voltage)
                # 计算电压越限率
                combined = np.concatenate((done_pv, done_storage))  # 合并两个数组
                count_ones = np.sum(combined)  # 统计值为 1 的数量（没有越限的智能体的数量）
                over_limit_rate = 1 - count_ones / len(combined)  # 计算越限率
                over_limit_rates.append(over_limit_rate)
                #更新网络参数
                self.update()

            print(f"Episode {episode + 1}, global_Reward: {self.global_reward}")

        return voltage_data, over_limit_rates

    def _initialize_replay_buffer(self, env, num_initial_steps=100, epsilon=1.0):
        """使用随机策略初始化缓冲区"""
        for _ in range(num_initial_steps):
            state_pv = env.reset_pv()
            state_storage = env.reset_storage()

            max_steps = 10
            step_count = 0
            while step_count < max_steps:
                step_count += 1
                actions_pv = [agent.get_action(state_pv[i], epsilon) for i, agent in enumerate(self.pv_agents)]
                actions_storage = [agent.get_action(state_storage[i], epsilon) for i, agent in
                                   enumerate(self.storage_agents)]

                next_state_pv, rewards_pv, done_pv = env.step_pv(actions_pv)
                next_state_storage, rewards_storage, done_storage = env.step_storage(actions_storage)

                for i, agent in enumerate(self.pv_agents):
                    self.pv_replay_buffer.add(state_pv[i], actions_pv[i], rewards_pv[i], next_state_pv[i], done_pv[i])
                for i, agent in enumerate(self.storage_agents):
                    self.storage_replay_buffer.add(state_storage[i], actions_storage[i], rewards_storage[i],
                                                   next_state_storage[i], done_storage[i])

                state_pv = next_state_pv
                state_storage = next_state_storage

    def save_model(self, directory):
        for i, agent in enumerate(self.pv_agents):
            torch.save(agent.policy_net.state_dict(), f"{directory}/pv_agent_{i}_policy.pth")
            torch.save(agent.value_net.state_dict(), f"{directory}/pv_agent_{i}_value.pth")

        for i, agent in enumerate(self.storage_agents):
            torch.save(agent.policy_net.state_dict(), f"{directory}/storage_agent_{i}_policy.pth")
            torch.save(agent.value_net.state_dict(), f"{directory}/storage_agent_{i}_value.pth")

    def load_model(self, directory):
        for i, agent in enumerate(self.pv_agents):
            agent.policy_net.load_state_dict(torch.load(f"{directory}/pv_agent_{i}_policy.pth"))
            agent.value_net.load_state_dict(torch.load(f"{directory}/pv_agent_{i}_value.pth"))

        for i, agent in enumerate(self.storage_agents):
            agent.policy_net.load_state_dict(torch.load(f"{directory}/storage_agent_{i}_policy.pth"))
            agent.value_net.load_state_dict(torch.load(f"{directory}/storage_agent_{i}_value.pth"))

    #TODO:这个函数是否有必要
    def online_train(self, num_steps, pp_net, pv_bus, es_bus):
        env = IEEE123bus(pp_net, pv_bus, es_bus)
        voltage_data = []
        over_limit_rates = []

        state_pv = env.reset_pv()
        state_storage = env.reset_storage()

        for step in range(num_steps):
            actions_pv = [agent.get_action(state_pv[i]) for i, agent in enumerate(self.pv_agents)]
            actions_storage = [agent.get_action(state_storage[i]) for i, agent in enumerate(self.storage_agents)]

            next_state_pv, rewards_pv, done_pv = env.step_pv(actions_pv)
            next_state_storage, rewards_storage, done_storage = env.step_storage(actions_storage)

            for i, agent in enumerate(self.pv_agents):
                self.pv_replay_buffer.add(state_pv[i], actions_pv[i], rewards_pv[i], next_state_pv[i], done_pv[i])
            for i, agent in enumerate(self.storage_agents):
                self.storage_replay_buffer.add(state_storage[i], actions_storage[i], rewards_storage[i],
                                               next_state_storage[i], done_storage[i])

            state_pv = next_state_pv
            state_storage = next_state_storage

            # 实时更新
            self.update()

            # 记录电压数据并计算电压越限率
            voltage = env.network.res_bus.vm_pu.to_numpy()
            voltage_data.append(voltage)
            # 计算电压越限率
            combined = np.concatenate((done_pv, done_storage))  # 合并两个数组
            count_ones = np.sum(combined)  # 统计值为 1 的数量（没有越限的智能体的数量）
            over_limit_rate = 1 - count_ones / len(combined)  # 计算越限率
            over_limit_rates.append(over_limit_rate)

            print(f"Step {step + 1}, Voltage Limit Exceedance Rate: {over_limit_rate}")
            # 定期保存模型
            # if (step + 1) % 100 == 0:
            #     self.save_model('online_model_directory')
        return voltage_data, over_limit_rates

