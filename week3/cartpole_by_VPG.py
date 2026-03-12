import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        # TODO: 定义网络层
        # 输入: obs_dim, 输出: act_dim (动作概率)
        pass
    
    def forward(self, obs):
        return self.net(obs)
        # TODO: 前向传播，输出 logits
        pass
    
    def get_action(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
        # TODO: 
        # 1. 调用 forward 得到 logits
        # 2. 用 Categorical 分布采样动作
        # 3. 返回动作和 log_prob (用于梯度计算)
        pass

class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1)

def train_vpg_baseline(env_name='CartPole-v1', episodes=1000, 
                       policy_lr=1e-3, value_lr=1e-2, gamma=0.99):
    env = gym.make(env_name)
    print(env.spec.max_episode_steps)  # 应该输出 500
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = Policy(obs_dim, act_dim)
    value_fn = ValueFunction(obs_dim)
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_fn.parameters(), lr=value_lr)
    
    episode_rewards = []

    for ep in range(episodes):
        # 收集轨迹（同 Level 1）
        obs_list, action_list, reward_list, log_prob_list = [], [], [], []
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, log_prob = policy.get_action(torch.tensor(obs, dtype=torch.float32))
            next_obs, reward,terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            log_prob_list.append(log_prob)
            obs = next_obs
            # TODO: 收集轨迹，同时存储 obs 用于训练 V(s)
            pass
        
        # 计算回报
        returns = compute_returns(reward_list, gamma)
        torch_obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
        values = value_fn(torch_obs)
        loss_fn = nn.MSELoss()
        value_loss = loss_fn(values, returns)
        # TODO: 更新价值函数
        # 目标: V(s) 接近 G_t，用 MSE loss
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        with torch.no_grad():
             values = value_fn(torch_obs)
        advantage = returns - values  # Advantage = G_t - V(s)
        # TODO: 计算 Advantage = G_t - V(s)
        # 注意: 用 detach() 阻止梯度流到 value_fn
        log_probs = torch.stack(log_prob_list)
        policy_loss = -(log_probs * advantage.detach()).mean()  # 策略梯度损失
        # TODO: 更新策略，用 Advantage 代替 Return
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        total_reward = sum(reward_list)
        episode_rewards.append(total_reward)
        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep}, Avg Reward: {avg_reward:.1f}")
            #print(f"Episode {ep}, Reward: {sum(reward_list)}")
    return episode_rewards


def compute_returns(rewards, gamma=0.99):
    returns = []
    G_t = 0
    for t in reversed(range(len(rewards))):
        G_t = rewards[t] + gamma * G_t
        returns.insert(0, G_t)
    return torch.tensor(returns)
    # TODO: 计算折扣回报 G_t = r_t + gamma * r_{t+1} + ...
    # 输入: rewards 列表 [r0, r1, r2, ...]
    # 输出: returns 张量 [G0, G1, G2, ...]
    # 提示: 从后往前算
    pass

def train_vpg(env_name='CartPole-v1', episodes=1000, lr=1e-3, gamma=0.99):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = Policy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    episode_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs = []  # 存储 log π(a|s)
        rewards = []    # 存储奖励
        
        done = False
        while not done:
            action, log_prob = policy.get_action(torch.tensor(obs, dtype=torch.float32))  # 选动作并存储 log_prob
            next_obs, reward, done, _, _ = env.step(action)  # 执行动作，得到 next_obs, reward, done
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs  # 更新 obs
            # TODO: 收集一条轨迹
            # 1. 用 policy.get_action 选动作
            # 2. 执行动作，得到 next_obs, reward, done
            # 3. 存储 log_prob 和 reward
            # 4. obs = next_obs
            pass
        
        # TODO: 计算回报
        returns = compute_returns(rewards, gamma)
        loss = -(torch.stack(log_probs) * returns).mean()  # 策略梯度损失
        # TODO: 策略梯度更新
        # loss = - (log_prob * return).mean()
        # 注意: 这里不需要 backward 每一步，等整条轨迹收集完再更新
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        episode_rewards.append(sum(rewards))
        
        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep}, Avg Reward: {avg_reward:.1f}")
    
    return episode_rewards

if __name__ == "__main__":
    rewards = train_vpg_baseline()
    
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('VPG on CartPole')
    plt.savefig('vpg_training.png')