# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 04 数値微分
# Created by: Owner
# Created on: 2021/10/07
# Page      : P19 - P24
# ******************************************************************************


# ＜概要＞
# - 微分の基本的な方法である数値微分を確認する
#   --- 微分とは変化の割合を表したもの
#   --- 数値微分は中心差分近似という手法を用いて指定した点における関数の傾きを求める
#   --- バックプロパゲーションの検証用として重宝する（購買確認）


# ＜数値微分の問題点＞
# - 数値微分は誤差を含む（変化量の取り方に依存する）
# - 計算コストが大きい（ニューラルネットではパラメータが非常に多いため問題となる）


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 単純な数値微分
# 3 連結した関数の微分


# 0 準備 -------------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 クラス定義 ---------------------------------------------------------------

# クラス定義
# --- 変数を格納するクラス
class Variable:
    def __init__(self, data):
        self.data = data

# クラス定義
# --- 関数を実行するための変数を抽出/格納してメイン計算に渡す基底クラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

# クラス継承
# --- Square計算
class Square(Function):
    def forward(self, x):
        return x ** 2

# クラス継承
# --- Exp計算
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

# 関数定義
# --- 数値微分
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# 2 単純な数値微分 -----------------------------------------------------------

# ＜ポイント＞
# - 数値微分を用いると導関数の計算とほぼ同じ値を得ることができる
#   --- 誤差を含む（今後の課題）


# インスタンス生成
# --- 関数
# --- 数値格納
f = Square()
x = Variable(np.array(2.0))

# 数値微分
# --- ｢y = x^2｣の｢2｣における微分
dy = numerical_diff(f, x)

# 確認
print(dy)


# 3 連結した関数の微分 -------------------------------------------------------

# ＜ポイント＞
# - Step3で作成した｢関数の連結｣で合成関数を生成して数値微分を適用する
# - 数値微分の関数を定義することで、関数が複雑になっても同じように数値微分を計算することができる
#   --- 関数(f)を引数として関数に渡すのがポイント（高階関数）


# 関数定義
# --- 連結した関数(合成関数)
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

# インスタンス生成
# --- 数値格納
x = Variable(np.array(0.5))

# 数値微分
dy = numerical_diff(f, x)

# 確認
print(dy)
