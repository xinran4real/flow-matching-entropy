import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class VectorField(nn.Module):
    """
    简单的 MLP 速度场 v(x, t)
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(), # Tanh 比较平滑，适合拟合流场
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.repeat(x.shape[0], 1)
        elif t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
        x_t = torch.cat([x, t], dim=1)
        return self.net(x_t)

def get_theoretical_entropy(dist_type, params):
    """计算理论熵"""
    if dist_type == 'gaussian':
        # H = 1 + log(2pi) + 2*log(sigma) (对于2D各向同性高斯)
        sigma = params['sigma']
        return 1 + math.log(2 * math.pi) + 2 * math.log(sigma)
    elif dist_type == 'uniform':
        # H = log(Area)
        scale = params['scale']
        area = (2 * scale) ** 2
        return math.log(area)
    return 0.0

def sample_target(dist_type, batch_size, params):
    if dist_type == 'gaussian':
        return torch.randn(batch_size, 2) * params['sigma']
    elif dist_type == 'uniform':
        return (torch.rand(batch_size, 2) - 0.5) * 2 * params['scale']

def compute_divergence_finite_difference(model, x, t, h=1e-3):
    """
    使用有限差分 + Hutchinson 估计器计算散度。
    完全不需要 autograd，显存开销极小。
    """
    # 1. 采样 Rademacher 噪声向量 epsilon (值域 {-1, 1})
    # 这种噪声比高斯噪声方差更小，收敛更快
    epsilon = torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1
    
    # 2. 计算当前位置的速度 v(x)
    v_x = model(x, t)
    
    # 3. 计算扰动位置的速度 v(x + h * epsilon)
    x_perturbed = x + h * epsilon
    v_x_perturbed = model(x_perturbed, t)
    
    # 4. 有限差分近似 Jacobian-Vector Product (J * epsilon)
    # J * epsilon ≈ (v(x+h*e) - v(x)) / h
    jvp_approx = (v_x_perturbed - v_x) / h
    
    # 5. Hutchinson Trace 估计: epsilon^T * (J * epsilon)
    # 逐元素相乘后在维度方向求和
    divergence = (epsilon * jvp_approx).sum(dim=1, keepdim=True)
    
    return divergence, v_x

@torch.no_grad()
def solve_ode_and_measure_entropy(model, batch_size=5000, steps=100, device='cpu'):
    """
    求解 ODE 并通过有限差分追踪对数密度
    """
    model.eval()
    
    # --- 1. 初始分布 p0 (标准正态) ---
    x = torch.randn(batch_size, 2).to(device)
    
    # log p0(x) = -d/2 log(2pi) - 0.5 * ||x||^2
    d = 2
    log_p = -0.5 * d * math.log(2 * math.pi) - 0.5 * (x**2).sum(dim=1, keepdim=True)
    
    dt = 1.0 / steps
    
    print(f"开始 ODE 采样 (有限差分模式, Steps={steps})...")
    
    for i in range(steps):
        t_val = i / steps
        t = torch.full((batch_size, 1), t_val).to(device)
        
        # --- 2. 计算速度和散度 (Finite Difference) ---
        # 这一步完全没有梯度图开销
        div, v = compute_divergence_finite_difference(model, x, t, h=1e-3)
        
        # --- 3. 欧拉积分更新 ---
        # 更新位置: x(t+dt) = x(t) + v * dt
        x = x + v * dt
        
        # 更新对数密度: log_p(t+dt) = log_p(t) - div * dt
        log_p = log_p - div * dt
        
    # --- 4. 计算熵 ---
    # H(p) = -E[log p(x)]
    estimated_entropy = -log_p.mean().item()
    
    return x.cpu(), estimated_entropy

def train(dist_type, params, iterations=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VectorField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nTraining {dist_type}...")
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        batch_size = 512
        
        # Conditional Flow Matching Loss
        x0 = torch.randn(batch_size, 2).to(device)
        x1 = sample_target(dist_type, batch_size, params).to(device)
        t = torch.rand(batch_size, 1).to(device)
        
        # 线性插值路径
        x_t = (1 - t) * x0 + t * x1
        target_v = x1 - x0 # 目标速度
        
        pred_v = model(x_t, t)
        loss = (pred_v - target_v).pow(2).mean()
        
        loss.backward()
        optimizer.step()
        
    return model

def main():
    configs = [
        {'name': 'Gaussian (Sigma=0.5)', 'type': 'gaussian', 'params': {'sigma': 0.5}},
        {'name': 'Gaussian (Sigma=2.0)', 'type': 'gaussian', 'params': {'sigma': 2.0}},
        {'name': 'Uniform (Scale=2.0)', 'type': 'uniform', 'params': {'scale': 2.0}},
    ]
    
    plt.figure(figsize=(15, 5))
    
    for idx, conf in enumerate(configs):
        # 1. 训练
        model = train(conf['type'], conf['params'])
        
        # 2. 测量熵 (使用有限差分)
        final_x, model_entropy = solve_ode_and_measure_entropy(
            model, batch_size=5000, steps=100, device=next(model.parameters()).device
        )
        
        # 3. 理论值对比
        theo_entropy = get_theoretical_entropy(conf['type'], conf['params'])
        err = abs(model_entropy - theo_entropy)
        
        print(f"[{conf['name']}]")
        print(f"  理论熵: {theo_entropy:.4f}")
        print(f"  模型熵: {model_entropy:.4f} (Finite Diff)")
        print(f"  误差:   {err:.4f}")
        
        # 4. 绘图
        plt.subplot(1, 3, idx+1)
        plt.scatter(final_x[:, 0], final_x[:, 1], s=1, alpha=0.3, label='Samples')
        if conf['type'] == 'uniform':
            s = conf['params']['scale']
            plt.plot([-s, s, s, -s, -s], [-s, -s, s, s, -s], 'r--', lw=2, label='Boundary')
        plt.title(f"{conf['name']}\nTheory: {theo_entropy:.2f} | Model: {model_entropy:.2f}")
        plt.legend()
        plt.axis('equal')
        
    plt.tight_layout()
    plt.savefig("validate_algorithm.png")

if __name__ == "__main__":
    main()