# ppo_fill_in_the_blank.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """Actor-Critic 网络，输出策略分布和状态价值"""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        
        # TODO: 定义共享特征层 (shared feature extraction)
        # 建议: 2层 MLP, 输入 obs_dim, 输出 hidden_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            # 填空 1
        )
        
        # TODO: Actor 头，输出动作 logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, act_dim)
            # 填空 2
        )
        
        # TODO: Critic 头，输出状态价值 V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
            # 填空 3
        )
    
    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
        """
        前向传播
        返回: (logits, value)
        """
        # TODO: 填空 4
        pass
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        action_dist = Categorical(logits=logits)
        if action is None:
            action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action, log_prob, entropy, value
        """
        采样动作，并返回相关信息用于训练
        如果 action 为 None，则采样新动作；否则计算给定 action 的 log_prob
        
        返回: (action, log_prob, entropy, value)
        """
        # TODO: 填空 5
        # 1. 调用 forward 得到 logits 和 value
        # 2. 创建 Categorical 分布
        # 3. 如果 action 为 None: 采样，否则用给定的 action
        # 4. 计算 log_prob 和 entropy
        pass


class RolloutBuffer:
    """存储一条轨迹的数据，用于 PPO 更新"""
    
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        """添加一步数据"""
        # TODO: 填空 6
        pass
    
    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        """清空缓冲区"""
        # TODO: 填空 7
        pass
    
    def get(self):
        """返回所有数据为张量"""
        return (
            torch.FloatTensor(np.array(self.obs)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.dones),
            torch.FloatTensor(self.values),
        )


class PPO:
    """PPO 算法实现"""
    
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,           # GAE 参数
        clip_eps=0.2,       # 裁剪阈值 ε
        K_epochs=10,        # 每次数据复用次数
        batch_size=64,
        value_coef=0.5,     # 价值损失系数
        entropy_coef=0.01,  # 熵正则系数
        max_grad_norm=0.5,  # 梯度裁剪阈值
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络和优化器
        self.ac = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer()
    
    def compute_gae(self, rewards, values, dones):
        """
        计算 GAE (Generalized Advantage Estimation)
        
        参数:
            rewards: [T] 每步奖励
            values: [T+1] 每步价值估计（包含最后一步的 V(s')）
            dones: [T] 是否结束标志
        
        返回:
            advantages: [T] 优势估计
            returns: [T] TD(λ) 回报
        
        公式:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
            从后往前递推: A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        T_reverse = reversed(range(T))
        gae = 0
        # TODO: 填空 8 - 实现 GAE 计算
        # 提示:
        # 1. 从后往前遍历 (reversed(range(T)))
        # 2. 计算 TD 误差 delta
        # 3. 递推计算 advantage
        # 4. 最后 advantages[t] = gae
        # 5. returns = advantages + values[:-1] (去掉最后一个 next_value)
        
        pass  # 删除这行，填入你的代码
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self):
        """
        使用收集的数据更新策略
        """
        # 获取数据
        obs, actions, old_log_probs, rewards, dones, values = self.buffer.get()
        
        # 添加最后一步的 value 用于 GAE 计算
        with torch.no_grad():
            _, _, _, last_value = self.ac.get_action_and_value(obs[-1:])
            values = torch.cat([values, last_value])
        
        # 计算 GAE 和 returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # TODO: 填空 9 - 标准化优势
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_log_probs = old_log_probs  # 旧策略的 log_prob，不需要梯度
        
        # 多 epoch 更新
        update_losses = []
        
        for epoch in range(self.K_epochs):
            # 生成随机索引，小批量训练
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # TODO: 填空 10 - 前向传播获取新策略的 log_prob 和价值
                # _, new_log_probs, entropy, new_values = self.ac.get_action_and_value(...)
                
                # TODO: 填空 11 - 计算 ratio = exp(new_log_prob - old_log_prob)
                # ratio = torch.exp(...)
                
                # TODO: 填空 12 - Clipped Surrogate Objective
                # surr1 = ratio * mb_advantages
                # surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                # actor_loss = -torch.min(surr1, surr2).mean()
                
                # TODO: 填空 13 - Critic Loss (MSE)
                # critic_loss = F.mse_loss(new_values.squeeze(), mb_returns)
                
                # TODO: 填空 14 - 总 Loss
                # loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                # loss.backward()  # 取消注释
                # TODO: 填空 15 - 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # update_losses.append(loss.item())  # 取消注释
        
        # 清空缓冲区
        self.buffer.clear()
        
        return np.mean(update_losses) if update_losses else 0
    
    def select_action(self, obs, store=True):
        """
        选择动作
        store=True: 存储到 buffer（训练时）
        store=False: 仅推理（测试时）
        """
        with torch.no_grad() if not store else torch.enable_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, _, value = self.ac.get_action_and_value(obs_tensor)
            
            if store:
                return action.item(), log_prob.item(), value.item()
            else:
                return action.item()


def train_ppo(env_name='CartPole-v1', total_steps=200000, rollout_length=2048):
    """
    训练 PPO
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = PPO(obs_dim, act_dim)
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards = []
    
    step = 0
    updates = 0
    
    while step < total_steps:
        # 收集 rollout_length 步数据
        for _ in range(rollout_length):
            action, log_prob, value = agent.select_action(obs, store=True)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储到 buffer
            agent.buffer.add(obs, action, log_prob, reward, done, value)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, _ = env.reset()
        
        # 收集够数据，更新策略
        loss = agent.update()
        updates += 1
        
        if updates % 10 == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Step: {step}, Updates: {updates}, Avg Reward: {avg_reward:.1f}")
    
    return episode_rewards


def evaluate(agent, env_name='CartPole-v1', episodes=10):
    """评估训练好的策略"""
    env = gym.make(env_name)
    rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(obs, store=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    # 训练
    print("开始训练 PPO...")
    rewards = train_ppo(total_steps=100000)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO on CartPole-v1')
    plt.savefig('ppo_training.png')
    print("训练曲线保存为 ppo_training.png")
    
    # 评估
    # agent = PPO(...)  # 需要重新实例化或保存模型
    # mean_reward, std_reward = evaluate(agent)
    # print(f"评估结果: {mean_reward:.1f} +/- {std_reward:.1f}")