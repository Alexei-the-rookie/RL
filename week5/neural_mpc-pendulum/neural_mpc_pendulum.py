import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torchdiffeq import odeint
import matplotlib.pyplot as plt

class PendulumDynamicsModel(nn.Module):
    # 学习 Pendulum 的动力学
    def __init__(self):
        super().__init__()
        # 输入: [theta, theta_dot, u]，输出: [dtheta, dtheta_dot]
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, t, state_action):
        # state_action: [theta, theta_dot, u]
        # 注意：这里假设 u 包含在输入中
        return self.net(state_action)

class PendulumMPCController:
    # 基于学习的动力学模型进行 MPC 控制
    def __init__(self, model, horizon=20, n_samples=100, dt=0.05, top_k=100, action_dim=1):
        # model: 学习到的动力学模型
        # horizon: MPC 预测的时间步数
        # n_samples: 每次 MPC 优化时采样的动作序列数量
        # dt: 时间步长
        # top_k: MPC 优化时保留的最优动作序列数量
        # action_dim: 动作维度

        self.model = model
        self.model.eval()  # 确保模型在评估模式
        self.horizon = horizon
        self.n_samples = n_samples
        self.dt = dt
        self.top_k = top_k
        self.action_dim = action_dim

        # Pendulum 的动作范围
        self.action_min = -2.0  
        self.action_max = 2.0

    def predict_trajectory(self, state0, action_seq):
        # 使用学习的动力学模型预测轨迹
        # state0: 初始状态 [theta, theta_dot]
        # action_seq: 动作序列 [horizon, action_dim]
        state = [state0]
        J = 0.0
        lambda t, x: self.model(t, torch.cat([x, action_seq[int(t // self.dt)]], dim=-1))
        for step in range(self.horizon):
            next_state = odeint(lambda t, x: self.model(t, torch.cat([x, action_seq[step]], dim=-1)), state[-1], torch.tensor([0, self.dt]), method='rk4')[1]
            state.append(next_state)
        trajectory = torch.stack(state)  # [horizon+1, state_dim]
        for step in range(self.horizon):
            theta, theta_dot = trajectory[step]
            J = J + (theta ** 2 + 0.1 * theta_dot ** 2 + 0.01 * action_seq[step] ** 2)  # 代价函数
        return trajectory, J
    
    def get_random_action_by_random_shooting(self, state0, goal_state=None):
        # 使用随机射击方法获取 MPC 优化的动作
        # state0: 当前状态 [theta, theta_dot]
        # goal_state: 目标状态（Pendulum 的目标是 theta=0）
        action_seqs = torch.FloatTensor(self.n_samples, self.horizon, self.action_dim).uniform_(self.action_min, self.action_max)
        costs = []
        for i in range(self.n_samples):
            _, J = self.predict_trajectory(state0, action_seqs[i])
            costs.append(J.item())
        costs = np.array(costs)
        topk_indices = np.argsort(costs)[:self.top_k]
        best_action_seq = action_seqs[topk_indices[0]]  # 选择最优的动作序列
        return best_action_seq[0].numpy()  # 返回第一个动作
    
    def get_action_cem(self, state0, goal_state=None):
        # 使用交叉熵方法获取 MPC 优化的动作
        action_seqs = torch.FloatTensor(self.n_samples, self.horizon, self.action_dim).uniform_(self.action_min, self.action_max)
        costs = []

def collect_pendulum_data(n_episodes=50, max_steps=200):
    """用随机策略或训练好的 PPO 收集数据"""
    env = gym.make('Pendulum-v1')
    
    # 可选：先用 PPO 训练一个策略来收集高质量数据
    # model = PPO('MlpPolicy', env).learn(10000)
    
    data = []
    for _ in range(n_episodes):
        obs, _ = env.reset()  # obs = [cos(theta), sin(theta), theta_dot]
        trajectory = []
        
        for _ in range(max_steps):
            # 随机动作
            action = env.action_space.sample()
            
            # 转换 obs 到 [theta, theta_dot]（从 sin/cos 还原）
            cos_theta, sin_theta, theta_dot = obs
            theta = np.arctan2(sin_theta, cos_theta)
            
            trajectory.append({
                'state': np.array([theta, theta_dot]),
                'action': action,
                'next_state': None  # 下一步填充
            })
            
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # 计算 next_theta
            next_cos, next_sin, next_theta_dot = next_obs
            next_theta = np.arctan2(next_sin, next_cos)
            
            trajectory[-1]['next_state'] = np.array([next_theta, next_theta_dot])
            
            obs = next_obs
            if terminated or truncated:
                break
        
        data.extend(trajectory)
    
    env.close()
    return data

def train_pendulum_model():
    print("收集数据...")
    data = collect_pendulum_data(n_episodes=100)
    
    # 准备张量
    states = torch.FloatTensor(np.array([d['state'] for d in data]))  # [N, 2]
    actions = torch.FloatTensor(np.array([d['action'] for d in data]))  # [N, 1]
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    elif actions.size(-1) != 1:
        actions = actions.view(-1, 1)
    next_states = torch.FloatTensor(np.array([d['next_state'] for d in data]))  # [N, 2]
    
    # 计算状态导数（数值微分近似）
    dt = 0.05  # Pendulum 默认 dt
    state_derivatives = (next_states - states) / dt  # [N, 2]
    
    model = PendulumDynamicsModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("训练模型...")
    for epoch in range(1500):
        # 构建输入 [theta, theta_dot, u]
        inputs = torch.cat([states, actions], dim=-1)
        
        # 预测导数
        pred_derivatives = model(None, inputs)  # t 参数这里不需要
        
        # 损失：预测的导数应该接近数值计算的导数
        loss = torch.mean((pred_derivatives - state_derivatives) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # 测试：用学习模型模拟一段轨迹
    test_initial = torch.tensor([[0.0, 0.0]])  # 初始向下，角速度 0
    test_action = torch.tensor([[1.0]])  # 恒定扭矩
    
    # 模拟 5 秒
    t = torch.linspace(0, 5, 100)
    
    def dynamics(t, x):
        # 拼接动作（这里假设恒定动作）
        xu = torch.cat([x, test_action.expand(x.size(0), -1)], dim=-1)
        return model(t, xu)
    
    with torch.no_grad():
        trajectory = odeint(dynamics, test_initial, t, method='rk4')
        trajectory = trajectory.squeeze(1).numpy()
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t.numpy(), trajectory[:, 0])
    plt.xlabel('Time')
    plt.ylabel('Theta')
    plt.title('Learned Model: Theta vs Time')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t.numpy(), trajectory[:, 1])
    plt.xlabel('Time')
    plt.ylabel('Theta_dot')
    plt.title('Learned Model: Velocity vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learned_pendulum4mpc.png')
    print("结果保存为 learned_pendulum4mpc.png")

    return model



if __name__ == "__main__":
    train_pendulum_model()