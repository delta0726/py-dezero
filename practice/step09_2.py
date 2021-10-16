# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 09 関数をより便利に（Backwardメソッドの簡略化）
# Created by: Owner
# Created on: 2021/10/16
# Page      : P51 - P52
# ******************************************************************************


# ＜概要＞
# - gradに毎回1を追加する手間を省く


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 バックプロパゲーションの実行


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 クラス定義 ------------------------------------------------------------

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 追加
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


# 2 バックプロパゲーションの実行 -------------------------------------------

# インスタンス生成
# --- 数値格納
# --- 関数の格納
x = Variable(np.array(0.5))
y = square(exp(square(x)))

# 逆伝播
# y.grad = np.array(1.0)
y.backward()
print(x.grad)
