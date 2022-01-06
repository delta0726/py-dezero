# ******************************************************************************
# Title       : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage       : 1 微分を自動で求める
# Step        : 02 変数を生み出す関数
# Create Date : 2021/10/04
# Update Date : 2022/01/05
# Page        : P9 - P13
# ******************************************************************************


# ＜概要＞
# - Functionクラスを関数用のデータ取得の基底クラスして、実際の計算はFunctionクラスを継承して実装するように修正する


# ＜参考＞
# raise NotImplementedError()の意味【継承】
# https://teratail.com/questions/259292


# ＜目次＞
# 0 準備
# 1 これまでの実装
# 2 Functionクラスの計算部分を分離して実装
# 3 Functionクラスを使う


# 0 準備 --------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 これまでの実装 ------------------------------------------------------

# クラス定義
# --- 実際のデータを変数として格納する
# --- Step1で実装済
class Variable:
    def __init__(self, data):
        self.data = data


# 2 Functionクラスの計算部分を分離して実装 ---------------------------------------

# ＜ポイント＞
# - forward()はFunctionクラスを継承した子クラスで実装することを前提とする
#   --- NotImplementedErrorでその意思を示すとともに、実装がない場合にエラーを返す


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
# --- Square関数の実装
# --- 基底クラス(Function)を継承して入力された値を二乗するメソッドを実装する
class Square(Function):
    def forward(self, x):
        return x ** 2


# 3 Functionクラスを使う ----------------------------------------------

# ＜ポイント＞
# - Functionクラスを継承したSquareクラスのインスタンスを生成する
# - デバッガーを用いてフローを確認！


# インスタンス生成
# --- fにFunction()ではなく、Square()を格納している点に注意
x = Variable(np.array(10))
f = Square()

# 関数実行
y = f(x)

# 確認
print(type(y))
print(y.data)
