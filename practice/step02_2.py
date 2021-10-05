# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 02 変数を生み出す関数
# Created by: Owner
# Created on: 2021/10/04
# Page      : P9 - P13
# ******************************************************************************


# ＜概要＞
# - Functionクラスを関数用のデータ取得の基底クラスして、実際の計算はFunctionクラスを継承して実装するように修正する


# ＜参考＞
# raise NotImplementedError()の意味【継承】
# https://teratail.com/questions/259292


# ＜目次＞
# 0 準備
# 1 関数をクラスとして実装


# 0 準備 --------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 関数をクラスとして実装 -----------------------------------------------


# クラス定義
# --- データを格納するクラス
class Variable:
    def __init__(self, data):
        self.data = data


# クラス定義
# --- Function: Variableからデータを取り出して、計算結果をVariableに格納する
# --- forward : 例外を発生させることでforwardメソッドが継承して実装していることをアピール
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError


# クラス継承
# --- 基底クラス(Function)を継承して入力された値を二乗するメソッドを実装する
class Square(Function):
    def forward(self, x):
        return x ** 2


# インスタンス生成
x = Variable(np.array(10))
f = Square()

# 関数実行
y = f(x)

# 確認
print(type(y))
print(y.data)
