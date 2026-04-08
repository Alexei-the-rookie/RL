# exercise_3_learn_pendulum.py
import gymnasium as gym
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class PendulumDynamicsModel(nn.Module):
    """学习 Pendulum 的动力学"""
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
        """
        state_action: [theta, theta_dot, u]
        注意：这里假设 u 是常数或包含在输入中
        """
        return self.net(state_action)

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
    plt.savefig('learned_pendulum.png')
    print("结果保存为 learned_pendulum.png")
    
    return model

if __name__ == "__main__":
    train_pendulum_model()