# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 07 バックプロパゲーションの自動化
# Created by: Owner
# Created on: 2021/10/09
# Page      : P37 - P41
# ******************************************************************************


# ＜概要＞
# - VariableクラスとFunctionクラスを拡張してバックプロパゲーションで微分が求められるようにする
#   --- クラスで出力値を出すと同時に、元の値を格納しているのがポイント（Define-by-Run）


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 逆伝播の自動化のために


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

# 追加
# --- output.set_creator(self)    # 出力変数に生みの親を覚えさせる
# --- self.output = output        # 出力も覚える
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


# 2 逆伝播の自動化のために ---------------------------------------------

# ＜ポイント＞
# - P34のグラフに沿って順伝播と逆伝播を行う
#   --- プロセスが分かりにくいのでデバッガーでの確認が必須


# インスタンス生成
# --- 関数（継承クラス）
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


# 3 動作テスト --------------------------------------------------------------

# ＜assert文＞
# - Pythonのassert文は条件をテストするデバッグ支援ツールです。
#   --- アサーションの条件がTrueの場合は何も起きず、プログラムは何事もなく動作し続けます
#   --- アサーションの条件がFalseと評価された場合はAssertionError例外が送出され、必要に応じてエラーメッセージが生成される

# ＜参考＞
# Pythonで本当に役立つ機能「アサーション」の使い方を解説！『Pythonトリック』から
# - https://codezine.jp/article/detail/12179


# 逆向きに計算グラフのノードを辿る
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x
