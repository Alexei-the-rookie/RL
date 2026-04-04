# exercise_1_spiral.py
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

class SpiralDynamics(nn.Module):
    """定义螺旋动力学的神经网络"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
    
    def forward(self, t, y):
        
        """
        参数:
            t: 时间（标量），虽然这里没直接用，但 odeint 会传入
            y: 状态 [batch_size, 2]
        返回:
            dy/dt: [batch_size, 2]
        """
        return self.net(y)

# 生成真实螺旋数据（作为训练目标）
def generate_spiral_data(n_points=100):
    t = torch.linspace(0, 2*3.14159, n_points)
    true_y = torch.zeros(n_points, 2)
    true_y[:, 0] = torch.cos(t) * torch.exp(-0.1*t)  # x = cos(t)*e^(-0.1t)
    true_y[:, 1] = torch.sin(t) * torch.exp(-0.1*t)  # y = sin(t)*e^(-0.1t)
    return t, true_y

def train():
    # 初始化
    model = SpiralDynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    t, true_y = generate_spiral_data()
    
    # 初始状态
    y0 = true_y[0].unsqueeze(0)  # [1, 2]
    
    for epoch in range(1000):
        # 前向传播：从 y0 出发，模拟到时间 t
        pred_y = odeint(model, y0, t, method='dopri5')  # [T, 1, 2]
        pred_y_squeezed = pred_y.squeeze(1)  # [T, 2]
        # 计算损失（整条轨迹的 MSE）
        loss = torch.mean((pred_y_squeezed - true_y) ** 2)
        # 反向传播（自动使用伴随法）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # 可视化
    with torch.no_grad():
        pred_y = odeint(model, y0, t, method='dopri5')
        pred_y_squeezed = pred_y.squeeze(1)
    plt.figure(figsize=(6, 6))
    plt.plot(true_y[:, 0], true_y[:, 1], 'b-', label='True Spiral', linewidth=2)
    plt.plot(pred_y_squeezed[:, 0], pred_y_squeezed[:, 1], 'r--', label='Neural ODE Approximation', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Neural ODE Fitting Spiral Dynamics')
    plt.savefig('spiral_fit.png')
    print("结果保存为 spiral_fit.png")

if __name__ == "__main__":
    train()