# ******************************************************************************
# Title       : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage       : 1 微分を自動で求める
# Step        : 03 関数の連結
# Create Date : 2021/10/04
# Update Date : 2022/01/06
# Page        : P15 - P17
# ******************************************************************************


# ＜概要＞
# - 新しい関数をクラスとして実装して複数の関数を組み合わせて計算する
# - Variableインスタンスにデータを格納することで、メソッドチェーンを構築することができるようになる


# ＜目次＞
# 0 準備
# 1 これまでの実装
# 2 Exp関数の追加実装
# 3 動作確認


# 0 準備 ---------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 これまでの実装 ------------------------------------------------------

# クラス定義
# --- 実際のデータを変数として格納する
# --- Step1で実装済
class Variable:
    def __init__(self, data):
        self.data = data


# 2 Exp関数の追加実装 ------------------------------------------------------

# ＜ポイント＞
# - Functionクラスを継承してForwardメソッドにExp関数を実装する（Square関数と同様）


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
        raise NotImplementedError()


# クラス継承
# --- Square関数の実装
class Square(Function):
    def forward(self, x):
        return x ** 2

# クラス継承
# --- Exp関数の実装
class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# 3 動作確認 -------------------------------------------------------------

# ＜ポイント＞
# - __call__メソッドの入力と出力はいずれもVariableインスタンスとなっている
#   --- 具体的には、関数の適用前にVariableクラスからデータを取得して、関数の適用後にVariableクラスに再度格納している
#   --- これにより、DeZeroの関数を連続して使用することができる


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
# --- __call__メソッドの入力と出力はいずれもVariableインスタンス
# --- 関数の出力値の数値が直接されるわけではない点に注意
type(a)
type(b)
type(y)

# 出力値の確認
# --- 数値には.dataでアクセスする
print(a.data)
print(b.data)
print(y.data)
