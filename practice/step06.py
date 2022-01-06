# ******************************************************************************
# Title       : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage       : 1 微分を自動で求める
# Step        : 06 手作業によるバックプロパゲーション
# Create Date : 2021/10/08
# Update Date : 2022/01/07
# Page        : P31 - P35
# ******************************************************************************


# ＜概要＞
# - VariableクラスとFunctionクラスを拡張してバックプロパゲーションで微分が求められるようにする
#   --- クラスで出力値を出すと同時に、元の値を格納しているのがポイント


# ＜目次＞
# 0 準備
# 1 Variableクラスの追加実装
# 2 Functionクラスの追加実装
# 3 SquareとExpクラスの追加実装
# 4 バックプロパゲーションの実行


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 Variableクラスの追加実装 -----------------------------------------------

# ＜ポイント＞
# - 微分値を格納する変数(grad)を定義する
#   --- バックプロパゲーションに対応した仕組みを追加実装
#   --- gradはNoneで初期化し、逆伝播によってい実際に微分が計算されたときに値を格納


# クラス定義
# --- 変数を格納するクラス
# --- gradというインスタンス変数を追加（Noneで初期化）
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


# 2 Functionクラスの追加実装 -------------------------------------------------

# クラス定義
# --- 関数を実行するための変数を抽出/格納してメイン計算に渡す基底クラス
# --- 追加：inputをインスタンス変数に格納
# --- 追加：微分の計算を行う逆伝播の機能（backwardメソッド）
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


# 3 SquareとExpクラスの追加実装 -----------------------------------------

# ＜ポイント＞
# - 逆伝播のためのbackwardメソッドを追加する
#   --- Square()とExp()について微分公式に基づいて定義


# クラス継承
# --- Square計算
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

# クラス継承
# --- Exp計算
# --- backwardの場合は直前に格納しておいた値を呼び出して使用
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 4 バックプロパゲーションの実行 -------------------------------------------------

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
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
