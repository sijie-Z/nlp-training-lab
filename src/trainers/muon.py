"""
Muon Optimizer — Kimi K2 同款, 2× AdamW 的 token 效率

论文: KellerJordan/Muon (2025), Kimi K2 Technical Report (2026)
核心思想: 用 Newton-Schulz 迭代做矩阵正交化, 替代 Adam 的逐元素缩放

为什么 Muon 比 Adam 好?

Adam (2014):
    m = β1*m + (1-β1)*g       # 动量（逐元素）
    v = β2*v + (1-β2)*g²      # 二阶矩（逐元素）
    θ = θ - lr * m / √v       # 逐元素更新

Muon (2025-2026):
    m = β*m + (1-β)*g         # 矩阵动量（和 Adam 一样）
    m = m - (trace(m)/dim) * I  # 去迹（去除缩放偏差）
    m = NewtonSchulz(m)        # 正交化 → 等价于 SVD 的 UV^T
    θ = θ - lr * m             # 更新

关键差异:
    Adam 逐元素做 lr*grad — 每个参数独立缩放
    Muon 矩阵级做正交化 — 保证权重更新的谱范数最优
    → 收敛快 ~2x（token 效率）, 但不改变最终性能上限


Newton-Schulz 迭代:

    目标: 计算 G / √(GG^T)（矩阵的极分解）
    等价于 SVD(G) = UΣV^T 中的 UV^T

    迭代公式（Quintic, 5 次迭代）:
        X = G / ‖G‖                               # 初始归一化
        for i in range(5):
            A = X @ X^T                            # 计算 Gram 矩阵
            B = b_i * A + c_i * (A @ A)            # quintic 多项式
            X = a_i * X + B @ X                    # 更新

    系数（KellerJordan 优化版）:
        a = [4.0848, 3.9505, 3.7418, 2.8769, 2.8366]
        b = [-6.8946, -6.3029, -5.5913, -3.1427, -3.0525]
        c = [2.9270, 2.6377, 2.3037, 1.2046, 1.2012]


MuonClip (Kimi K2 增强):

    Kimi K2 在 Muon 基础上加了 QK-Clip:
    当任意 attention head 的 max(logit) > τ (通常 100) 时,
    对该 head 的 Q/K 投影矩阵做权重层面的 rescale
    → 在 15.5T tokens 上零 loss spike


简化版 Muon（用于小模型）:
    1. 对每个 ≥2D 的参数矩阵（Linear.weight）用 Muon 更新
    2. 对 1D 参数（bias, norms）用 AdamW 更新
    3. Newton-Schulz 迭代 5 步
    4. 可选: QK-Clip（检测异常 logit 值）
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Newton-Schulz 迭代（Muon 的核心）
# ============================================================
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz 迭代：计算极分解 G = UP 中的 U

    等价于 SVD(G) = UΣV^T 中的 UV^T
    比 torch.svd 快 >10x（在 GPU 上）

    这是 KellerJordan/modded-nanogpt 的参考实现
    """
    assert len(G.shape) == 2

    # 预归一化: 保证 spectral norm ≤ 1
    G = G / (G.norm() + eps)

    # 系数（quintic polynomial）
    a = torch.tensor([4.0848, 3.9505, 3.7418, 2.8769, 2.8366],
                     device=G.device, dtype=torch.float32)
    b = torch.tensor([-6.8946, -6.3029, -5.5913, -3.1427, -3.0525],
                     device=G.device, dtype=torch.float32)
    c = torch.tensor([2.9270, 2.6377, 2.3037, 1.2046, 1.2012],
                     device=G.device, dtype=torch.float32)

    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T

    for i in range(steps):
        A = X @ X.T
        B = b[i] * A + c[i] * (A @ A)
        X = a[i] * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon 优化器 — Kimi K2 同款

    用法:
        # Linear 层用 Muon, 其他用 AdamW
        muon_params = [p for n, p in model.named_parameters()
                       if p.ndim >= 2 and 'lm_head' not in n]
        adam_params = [p for n, p in model.named_parameters()
                       if p.ndim < 2 or 'lm_head' in n]
        optimizer = Muon([
            {'params': muon_params, 'lr': muon_lr, 'use_muon': True},
            {'params': adam_params, 'lr': adam_lr, 'use_muon': False},
        ])

    Kimi K2 的超参:
        muon_lr = 0.02 (远高于 Adam)
        adam_lr = 0.002
        momentum = 0.95
        nesterov = True
        ns_steps = 5
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 nesterov=True, ns_steps=5, adam_lr_for_1d=0.002):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       nesterov=nesterov, ns_steps=ns_steps,
                       adam_lr_for_1d=adam_lr_for_1d)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                use_muon = group.get('use_muon', True)

                # Weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                if use_muon and grad.ndim >= 2:
                    # Muon 更新（矩阵参数）
                    self._muon_update(p, grad, lr, momentum, nesterov, ns_steps)
                else:
                    # AdamW 更新（1D 参数或 lm_head）
                    self._adam_update(p, grad, lr, momentum)

        return loss

    def _muon_update(self, p, grad, lr, momentum, nesterov, ns_steps):
        """Muon 核心更新: 动量 → 去迹 → 正交化 → 更新"""
        state = self.state[p]

        # 初始化动量 buffer
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(grad)

        buf = state['momentum_buffer']

        # 更新动量
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)

        if nesterov:
            # Nesterov: 用更新后的动量方向
            update = grad.add(buf, alpha=momentum)
        else:
            update = buf

        # 去迹（去除缩放偏差 — 矩阵越大, 迹越大, 更新越要缩小）
        update = self._de_trace(update)

        # Newton-Schulz 正交化
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)

        # 缩放学习率到合理的范围
        # Muon 的 lr 通常远高于 Adam (0.02 vs 0.002)
        # 通过除以矩阵规模的平方根来补偿
        scale = max(update.shape) ** 0.5
        p.add_(update, alpha=-lr * 0.2 / scale)

    def _adam_update(self, p, grad, lr, momentum):
        """简化版 Adam 更新（用于 1D 参数）"""
        state = self.state[p]

        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(grad)
            state['exp_avg_sq'] = torch.zeros_like(grad)
            state['step'] = 0

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        state['step'] += 1

        beta1 = momentum
        beta2 = 0.999

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
        step_size = lr / bias_correction1

        p.addcdiv_(exp_avg, denom, value=-step_size)

    @staticmethod
    def _de_trace(g):
        """去除矩阵的迹分量"""
        # 如果矩阵是方阵, 减去 scaled identity
        # 这对应于: G = G - (trace(G) / dim) * I
        if g.size(0) == g.size(1):
            trace = torch.trace(g)
            n = g.size(0)
            g = g - (trace / n) * torch.eye(n, device=g.device, dtype=g.dtype)
        return g


def create_muon_optimizer(model, muon_lr=0.02, adam_lr=0.002,
                          momentum=0.95, weight_decay=0.01,
                          exclude_keywords=('lm_head', 'bias', 'norm', 'wte')):
    """
    创建 Muon 优化器（Muon for weights, AdamW for 1D/excluded）

    自动分离:
        - Muon 参数: ≥2D 且不含排除关键词
        - Adam 参数: 1D 或含排除关键词

    Args:
        model: 模型实例
        muon_lr: Muon 学习率 (通常 0.01-0.02)
        adam_lr: AdamW 学习率 (通常 0.001-0.003)
        momentum: Muon 动量 (通常 0.95)
        weight_decay: 权重衰减
        exclude_keywords: 排除的关键词列表

    Returns:
        Muon optimizer instance
    """
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        excluded = any(kw in name for kw in exclude_keywords)
        is_matrix = p.ndim >= 2

        if is_matrix and not excluded:
            muon_params.append(p)
        else:
            adam_params.append(p)

    param_groups = []
    if muon_params:
        param_groups.append({
            'params': muon_params,
            'lr': muon_lr,
            'use_muon': True,
            'momentum': momentum,
            'weight_decay': weight_decay,
        })
    if adam_params:
        param_groups.append({
            'params': adam_params,
            'lr': adam_lr,
            'use_muon': False,
            'momentum': 0.9,  # AdamW 用标准 β1
            'weight_decay': weight_decay,
        })

    optimizer = Muon(param_groups, lr=muon_lr, momentum=momentum,
                     weight_decay=weight_decay)

    n_muon = len(muon_params)
    n_adam = len(adam_params)
    print(f"[Muon] {n_muon} 个参数组用 Muon, {n_adam} 个用 AdamW")
    print(f"[Muon] muon_lr={muon_lr}, adam_lr={adam_lr}, momentum={momentum}")

    return optimizer
