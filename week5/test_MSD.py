# exercise_2_mass_spring_damper.py
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

class RealMassSpringDamper:
    """真实的质量-弹簧-阻尼系统: m*x'' + c*x' + k*x = u"""
    def __init__(self, m=1.0, c=0.5, k=2.0):
        self.m = m
        self.c = c
        self.k = k
    
    def dynamics(self, t, state, u=0):
        """
        state: [x, v] 位置和速度
        返回: [v, a] 速度和加速度
        """
        x, v = state[..., 0], state[..., 1]
        a = (u - self.c * v - self.k * x) / self.m
        return torch.stack([v, a], dim=-1)

class NeuralMSD(nn.Module):
    """神经网络学习 MSD 动力学"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 输入: [x, v, u]，输出: [dx/dt, dv/dt]
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        
        # 初始化：让网络初始输出接近 0，避免一开始就发散
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, state, u=0):
        """
        state: [batch, 2] -> [x, v]
        u: 控制输入（标量或 [batch, 1]）
        """
        if not isinstance(u, torch.Tensor):
            u = torch.tensor([[u]], dtype=state.dtype, device=state.device)
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        
        # 拼接 [x, v, u]
        xvu = torch.cat([state, u.expand(state.size(0), 1)], dim=-1)
        return self.net(xvu)

def collect_training_data(real_system, n_traj=50, t_span=5.0, dt=0.01):
    """收集真实系统轨迹作为训练数据"""
    data = []
    t = torch.arange(0, t_span, dt)
    
    for _ in range(n_traj):
        # 随机初始条件
        x0 = torch.randn(1, 2) * 2  # x 和 v 随机初始化
        
        # 随机控制输入（正弦叠加）
        u_func = lambda t: 2.0 * torch.sin(2 * 3.14159 * t * (0.5 + torch.rand(1)))
        
        # 模拟真实轨迹
        trajectory = []
        state = x0
        for t_i in t:
            u_i = u_func(t_i)
            trajectory.append(state)
            # 欧拉法积分一步（真实系统）
            state = state + real_system.dynamics(t_i, state, u_i) * dt
        
        trajectory = torch.cat(trajectory, dim=0)  # [T, 2]
        
        # 存储 (初始状态, 控制序列, 轨迹)
        data.append({
            't': t,
            'x0': x0,
            'u': torch.stack([u_func(ti) for ti in t]),
            'trajectory': trajectory
        })
    
    return data

def train_neural_msd():
    # 真实系统
    real_sys = RealMassSpringDamper(m=1.0, c=0.5, k=2.0)
    model = NeuralMSD()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("收集训练数据...")
    train_data = collect_training_data(real_sys, n_traj=100)
    
    print("开始训练...")
    for epoch in range(500):
        total_loss = 0
        
        for batch in train_data[:10]:  # 每次用 10 条轨迹
            t = batch['t']
            x0 = batch['x0']
            u_seq = batch['u']  # [T, 1]
            target_traj = batch['trajectory']  # [T, 2]
            
            # 定义带控制输入的动力学函数
            def dynamics_with_control(t, x):
                dt = 0.01
                idx = min(int(float(t) / dt), len(u_seq)-1)
                u = u_seq[idx]
                return model(t, x, u)
            
            # Neural ODE 前向
            pred_traj = odeint(dynamics_with_control, x0, t, method='rk4')
            pred_traj = pred_traj.squeeze(1)  # [T, 1, 2] -> [T, 2]
            
            # 损失
            loss = torch.mean((pred_traj - target_traj) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/10:.6f}')
    
    # 测试：对比真实系统和学习模型的轨迹
    test_x0 = torch.tensor([[2.0, 0.0]])  # 初始偏移 2，速度 0
    test_t = torch.linspace(0, 10, 1000)
    test_u = 0  # 无控制，自由衰减
    
    # 真实轨迹
    real_traj = [test_x0]
    state = test_x0
    for i in range(len(test_t)-1):
        dt = test_t[i+1] - test_t[i]
        state = state + real_sys.dynamics(test_t[i], state, test_u) * dt
        real_traj.append(state)
    real_traj = torch.cat(real_traj, dim=0).detach().numpy()
    
    # 学习轨迹
    with torch.no_grad():
        def test_dynamics(t, x):
            return model(t, x, test_u)
        learned_traj = odeint(test_dynamics, test_x0, test_t, method='rk4')
        learned_traj = learned_traj.squeeze(1).numpy()
    
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

if __name__ == "__main__":
    train_neural_msd()