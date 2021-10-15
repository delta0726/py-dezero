# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 08 再帰からループへ
# Created by: Owner
# Created on: 2021/10/16
# Page      : P45 - P47
# ******************************************************************************


# ＜概要＞
# - 前のステップでVariableクラスに実装したbackwardメソッドの処理の効率化を行う
#   --- 再帰処理をループ処理に変更する（拡張性を高めるため）


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

    # 変更
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()                 # 関数を取得
            x, y = f.input, f.output        # 関数の入出力を取得
            x.grad = f.backward(y.grad)     # backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)     # 1つ前の関数をリストに追加


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


# 2 バックプロパゲーションの実行 -------------------------------------------

# インスタンス生成
# --- 関数
A = Square()
B = Exp()
C = Square()

# インスタンス生成
# --- 数値格納
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)
