"""
LMS 核心算法 —— 最小均方自适应滤波器（Widrow-Hoff, 1960）

用途：把"用过去若干根收益预测下一根收益"建模成一个线性自适应滤波器，
权重按每一步的瞬时误差在线更新（随机梯度下降），无需批量训练、天然走窗。

本文件提供：
  · NLMS      —— 归一化最小均方滤波器（单个，抗输入尺度）
  · LmsEnsemble —— 多阶集成（不同抽头数捕捉不同时间尺度，取平均）

更新公式（NLMS，带泄漏）：
    y(k)   = wᵀ x(k)                      # 预测
    e(k)   = d(k) − y(k)                  # 误差
    w(k+1) = (1−leak)·w(k) + μ·e(k)·x(k) / (ε + ‖x(k)‖²)

μ 越小越稳但收敛慢（经验 1e-3 ~ 0.1）；leak 抑制权重漂移。
"""
from __future__ import annotations

import numpy as np


class NLMS:
    """归一化最小均方自适应滤波器。"""

    def __init__(self, order: int, mu: float = 0.05, eps: float = 1e-6, leak: float = 0.0) -> None:
        self.order = int(order)
        self.mu = float(mu)
        self.eps = float(eps)
        self.leak = float(leak)
        self.w = np.zeros(self.order, dtype=float)

    def predict(self, x: np.ndarray) -> float:
        """给定输入向量 x（长度=order，最新值在前），输出预测。"""
        return float(self.w @ x)

    def adapt(self, x: np.ndarray, d: float) -> float:
        """用真实目标 d 更新权重，返回本步误差 e。"""
        y = self.predict(x)
        e = d - y
        norm = self.eps + float(x @ x)
        self.w = (1.0 - self.leak) * self.w + self.mu * e * x / norm
        return e


class LmsEnsemble:
    """
    多阶 NLMS 集成：每个成员用各自阶数的最近收益做输入，预测取平均。
    不同阶数 = 不同记忆长度，等价于多时间尺度自适应预测。
    """

    def __init__(
        self,
        orders: tuple[int, ...] = (3, 5, 10, 20),
        mu: float = 0.05,
        eps: float = 1e-6,
        leak: float = 0.0,
    ) -> None:
        self.orders = tuple(int(o) for o in orders)
        self.max_order = max(self.orders)
        self.members = [NLMS(o, mu, eps, leak) for o in self.orders]

    def predict(self, recent: np.ndarray) -> float:
        """
        recent: 最近 max_order 个收益，时间升序（旧->新）。
        每个成员取自己阶数长度的尾部并反转为"最新在前"后预测，取平均。
        """
        preds = []
        for f in self.members:
            x = recent[-f.order:][::-1]
            preds.append(f.predict(x))
        return float(np.mean(preds))

    def adapt(self, recent: np.ndarray, d: float) -> None:
        """所有成员用同一真实目标 d 各自更新。"""
        for f in self.members:
            x = recent[-f.order:][::-1]
            f.adapt(x, d)
