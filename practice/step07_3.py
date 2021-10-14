# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 07 バックプロパゲーションの自動化
# Created by: Owner
# Created on: 2021/10/09
# Page      : P43 - P44
# ******************************************************************************


# ＜概要＞
# - VariableクラスとFunctionクラスを拡張してバックプロパゲーションで微分が求められるようにする
#   --- クラスで出力値を出すと同時に、元の値を格納しているのがポイント（Define-by-Run）


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 バックプロパゲーションの実行


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 クラス定義 -----------------------------------------------------------

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None     # 追加

    # 追加
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)    # 追加：出力変数に生みの親を覚えさせる
        self.input = input
        self.output = output        # 追加：出力も覚える
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


# 2 バックプロパゲーションの実行 -------------------------------------------------

# ＜ポイント＞
# - P34のグラフに沿って順伝播と逆伝播を行う
#   --- プロセスが分かりにくいのでデバッガーでの確認が必須


# インスタンス生成
# --- 関数
A = Square()
B = Exp()
C = Square()

# インスタンス生成
# --- 数値格納
x = Variable(np.array(0.5))

# 順伝播
a = A(x)
b = B(a)
y = C(b)

# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)
