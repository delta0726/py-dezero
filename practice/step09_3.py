# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 09 関数をより便利に（ndarrayだけを扱う）
# Created by: Owner
# Created on: 2021/10/16
# Page      : P52 - P55
# ******************************************************************************


# ＜概要＞
# - Variableはndarrayだけを扱うように制限するための実装を行う


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 エラー動作の確認
# 3 変更に伴う副作用
# 4 バックプロパゲーションの実行


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 クラス定義 ------------------------------------------------------------

# 変更
# --- ndarray以外のデータを入力すると即座にエラーを返す（問題の早期発見）
# --- ただし、Noneは保持できるようにする
class Variable:
    def __init__(self, data):
        # 追加
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


# 追加
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))      # 変更
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


# 2 エラー動作の確認 ----------------------------------------------------

# OK
x = Variable(np.array(1.0))

# OK
x = Variable(None)

# Error
x = Variable(1.0)


# 3 変更に伴う副作用 ----------------------------------------------------

# ＜ポイント＞
# - 変数をスカラーで与えた場合にデータ型が変更されてしまう
# - as_array関数を導入するインセンティブ

# 通常のケース
# --- arrayをリストで作っている
x = np.array([1.0])
y = x ** 2
print(type(x), x.ndim)
print(type(y))

# 問題となるケース
# --- arrayをスカラーで作っている
# --- yの結果がfloatになってしまう
x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)
print(type(y))



# 4 バックプロパゲーションの実行 -------------------------------------------

# インスタンス生成
# --- 数値格納
# --- 関数の格納
x = Variable(np.array(0.5))
y = square(exp(square(x)))

# 逆伝播
# y.grad = np.array(1.0)
y.backward()
print(x.grad)

