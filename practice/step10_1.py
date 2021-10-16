# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 10 テストを行う（Pythonのユニットテスト）
# Created by: Owner
# Created on: 2021/10/16
# Page      : P57 - P59
# ******************************************************************************


# ＜概要＞
# - 小規模なユニットテストを体験する


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 ユニットテストの実行


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np
import unittest      # 追加


# 1 クラス定義 ------------------------------------------------------------

class Variable:
    def __init__(self, data):
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


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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


# 追加
# --- unittest.TestCaseを継承するクラスを実装
# --- テストケースは"test_*"で始まるメソッドとして定義する
# --- forward()の一致検証を行う
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)


# 2 ユニットテストの実行 ---------------------------------------------------

# ＜方法1＞
# ターミナルから以下のコマンドを実行
# python -m unittest practice/step10_1.py


# ＜方法2＞
# 以下のスクリプトのコメントを外してスクリプトを実行
# unittest.main()

# ターミナルから実行する場合
# python practice/step10_1.py

# 実行/デバッグ構成から実行する場合
# スクリプトパスに当該ファイルを指定（オプション指定不要）
