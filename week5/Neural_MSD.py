import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import random

class Mass_Spring_Damper(nn.Module):
    # 定义质量-弹簧-阻尼系统的神经网络
    def __init__(self,m=5.0, k=10.0, c=3.0):
        super().__init__()
        self.m = m  # 质量
        self.k = k  # 弹簧常数
        self.c = c  # 阻尼系数
    
    def Dynamics(self, t, state, u=0):
        """
        参数:
            t: 时间（标量）
            state: 状态 [batch_size, 2]，包含位置和速度
            u: 外部输入（标量），默认为0
        返回:
            dstate/dt: [batch_size, 2]，包含位置的导数（速度）和速度的导数（加速度）
        """
        x = state[:, 0]  # 位置
        v = state[:, 1]  # 速度

        a = (u - self.c * v - self.k * x) / self.m  # 加速度
        return torch.stack([v, a], dim=1)  # 返回位置的导数和速度的导数
    
class Neural_MSD(nn.Module):
    def __init__(self,input_size=3,hidden_size=64,output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, state, u=0):
        """
        参数:
            t: 时间（标量）
            state: 状态 [batch_size, 2]，包含位置和速度
            u: 外部输入标量，默认为0
        返回:
            dstate/dt: [batch_size, 2]，包含位置的导数（速度）和速度的导数（加速度）
        """
        if not isinstance(u, torch.Tensor):
            u = torch.tensor([[u]], dtype=state.dtype, device=state.device)
        if u.dim() == 1:
            u = u.unsqueeze(1)  # 确保 u 是 [batch_size, 1]
        input = torch.cat([state, u.expand(state.size(0), -1)], dim=1)  # [batch_size, 3]
        return self.net(input)  # 返回位置的导数和速度的导数
    
def collection(real_sys, n_traj=100, t_span=5.0, dt=0.01):
    data = []
    
    t = torch.arange(0, t_span, dt)
    for _ in range(n_traj):
        x_0 = torch.randn(1, 2) * 2
        u_func = lambda t: 2.0 * torch.sin(2 * 3.14159 * t * (0.5 + torch.rand(1)))  # 外部输入函数
        trajectories = []
        state = x_0
        for time in t:
            u = u_func(time)
            trajectories.append(state)
            state = state + real_sys.Dynamics(time, state, u) * dt  # Euler 方法更新状态
        trajectories = torch.cat(trajectories, dim=0)  # [T, 2]
        data.append(
            {
                't': t,
                'x_0': x_0,
                'input': torch.stack([u_func(time) for time in t]),
                'trajectory': trajectories,
                
            }
        )
    return data

def training():
    real_sys = Mass_Spring_Damper()
    model = Neural_MSD()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("收集训练数据...")
    train_data = collection(real_sys,n_traj=100)
    print("开始训练...")
    for epoch in range(500):
        total_loss = 0
        
        # 每次 Epoch 从 100 条轨迹中随机抽取 10 条作为一个 mini-batch
        sampled_data = random.sample(train_data, 10)
        for batch in sampled_data:
            t = batch['t']
            x_0 = batch['x_0']
            u_seq = batch["input"]
            target_traj = batch['trajectory']

            # 方案C：时间切片的多重打靶法（保留纯正 Neural ODE 的求解环节！）
            seq_len = 20  # 截断积分长度（例如每次让求解器推演 20 步），防止梯度消失
            start_idx = random.randint(0, len(t) - seq_len - 1)
            
            batch_t = t[start_idx : start_idx + seq_len]
            batch_x0 = target_traj[start_idx].unsqueeze(0)  # [1, 2] 作为这段积分离列的初始条件
            batch_target = target_traj[start_idx : start_idx + seq_len]
            
            # 在 ODE 积分内部封装网络调用
            def dynamics_with_input(t_val, state):
                dt = 0.01
                idx = min(int(float(t_val) / dt), len(u_seq)-1)
                u = u_seq[idx]
                return model(t_val, state, u)
            
            # Neural ODE 核心：让梯度直接穿过 odeint 积分器反向传播
            pred_traj = odeint(dynamics_with_input, batch_x0, batch_t, method='euler')
            pred_traj = pred_traj.squeeze(1)
            
            loss = torch.mean((pred_traj - batch_target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/10:.6f}')

    test_x0 = torch.tensor([[1.0, 0.0]])
    test_u_func = 0
    test_t = torch.linspace(0, 10, 1000)

    real_traj = [test_x0]
    state = test_x0
    for time in range(len(test_t)-1):
        dt = test_t[time+1] - test_t[time]
        state = state + real_sys.Dynamics(test_t[time], state, test_u_func) * dt
        real_traj.append(state)
    real_traj = torch.cat(real_traj, dim=0).detach().numpy()

    with torch.no_grad():
        def test_dynamics(t, state):
            return model(t, state, test_u_func)
        
        # 测试推理也使用 euler 保持一致
        learned_traj = odeint(test_dynamics, test_x0, test_t, method='euler')
        learned_traj = learned_traj.squeeze(1).detach().numpy()

    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(test_t.numpy(), real_traj[:, 0], 'b-', label='Real')
    plt.plot(test_t.numpy(), learned_traj[:, 0], 'r--', label='Learned')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.title('Position Comparison')
    
    plt.subplot(1, 3, 2)
    plt.plot(test_t.numpy(), real_traj[:, 1], 'b-', label='Real')
    plt.plot(test_t.numpy(), learned_traj[:, 1], 'r--', label='Learned')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.title('Velocity Comparison')
    
    plt.subplot(1, 3, 3)
    plt.plot(real_traj[:, 0], real_traj[:, 1], 'b-', label='Real Phase')
    plt.plot(learned_traj[:, 0], learned_traj[:, 1], 'r--', label='Learned Phase')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend()
    plt.title('Phase Portrait')
    
    plt.tight_layout()
    plt.savefig('msd_comparison.png')
    print("测试结果保存为 msd_comparison.png")

    # 记录绘制的所有数据到 txt 文件
    plot_data = np.column_stack((
        test_t.numpy(),
        real_traj[:, 0],
        real_traj[:, 1],
        learned_traj[:, 0],
        learned_traj[:, 1]
    ))
    np.savetxt('msd_plot_data.txt', plot_data, delimiter=',', header='Time, Real_Position, Real_Velocity, Learned_Position, Learned_Velocity', comments='')
    print("绘图数据保存为 msd_plot_data.txt")

if __name__ == "__main__":
    training()