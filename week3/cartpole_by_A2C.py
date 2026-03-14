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

class ActorCritic(nn.Module):
    """共享特征提取的 Actor-Critic"""
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, act_dim)    # 策略头
        self.critic = nn.Linear(hidden_dim, 1)          # 价值头
    
    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action_and_value(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()
        # TODO: 返回 action, log_prob, value
        pass

def train_a2c(env_name='CartPole-v1', total_steps=100000, 
              lr=1e-3, gamma=0.99):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    obs, _ = env.reset()
    episode_rewards = []
    episode_reward = 0
    for step in range(total_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action, log_prob, value = model.get_action_and_value(obs_tensor)
        next_obs, reward, done, _, _ = env.step(action)
        # TODO: 收集 n_steps 数据（或使用单步）
        # 存储: obs, action, reward, next_obs, done

        with torch.no_grad():
            if done:
                next_value = 0
            else:
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                _, next_value = model.forward(next_obs_tensor)
                next_value = next_value.squeeze().item()
            #_, _, next_value = model.get_action_and_value(torch.tensor(next_obs, dtype=torch.float32))
            #next_value =next_value.squeeze()

        target = reward + gamma * next_value * (1 - done)  # TD 目标
        target = torch.tensor(target, dtype=torch.float32)
        # TODO: 计算 TD 目标
        # target = r + gamma * V(s') * (1 - done)

        critic_loss = (value - target.detach()) ** 2  # MSE 损失
        # TODO: 计算 Critic loss (MSE)
        # critic_loss = (V(s) - target.detach())^2

        advantage = (target - value).detach()
        actor_loss = -log_prob * advantage  # 策略梯度损失
        # TODO: 计算 Actor loss (策略梯度)
        # advantage = (target - V(s)).detach()
        # actor_loss = -log_prob * advantage

        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: 总 loss = actor_loss + 0.5 * critic_loss
        # 反向传播，更新

        episode_reward += reward
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step % 100 == 0 and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            n_eps = min(100, len(episode_rewards))
            print(f"Step {step},Avg Reward (last {n_eps} eps): {avg_reward:.1f}")
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
    rewards = train_a2c()
    
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A2C on CartPole')
    plt.savefig('a2c_training.png')