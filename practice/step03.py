# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 03 関数の連結
# Created by: Owner
# Created on: 2021/10/06
# Page      : P15 - P17
# ******************************************************************************


# ＜概要＞
# - 新しい関数をクラスとして実装して複数の関数を組み合わせて計算する
# - Variableインスタンスにデータを格納することで、メソッドチェーンを構築することができるようになる


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 動作確認


# 0 準備 -------------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 クラス定義 --------------------------------------------------------------

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
        return output

    def foward(self, x):
        raise NotImplementedError()

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


# 2 動作確認 -------------------------------------------------------------

# インスタンス生成
# --- 関数定義
A = Square()
B = Exp()
C = Square()

# データ格納
x = Variable(np.array(0.5))

# 計算
a = A(x)
b = B(a)
y = C(b)

# データ型の確認
# --- いずれもVariableインスタンス
# --- 関数の出力値の数値が直接されるわけではない点に注意
type(a)
type(b)
type(y)

# 出力値の確認
# --- 数値には.dataでアクセスする
print(a.data)
print(b.data)
print(y.data)
