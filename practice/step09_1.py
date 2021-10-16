# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 09 関数をより便利に（Pythonの関数として利用）
# Created by: Owner
# Created on: 2021/10/16
# Page      : P49 - P51
# ******************************************************************************


# ＜概要＞
# - クラスとして関数を実装すると関数をアドホックに呼び出すには不都合なので改善を加える
#   --- ①インスタンス定義 ②関数実行 の2段階のステップが必要となる


# ＜目次＞
# 0 準備
# 1 クラス定義
# 2 クラスから関数へのアクセス方法
# 3 関数と同様の操作性を持たせる工夫
# 4 バックプロパゲーションの実行
# 5 バックプロパゲーションの関数を連続して実行


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import numpy as np


# 1 クラス定義 ------------------------------------------------------------

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 変更
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()                 # 関数を取得
            x, y = f.input, f.output        # 関数の入出力を取得
            x.grad = f.backward(y.grad)     # backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)     # 1つ前の関数をリストに追加


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

# 追加
def square(x):
    return Square()(x)

# 追加
def exp(x):
    return Exp()(x)


# 2 クラスから関数へのアクセス方法 -------------------------------------------

# ＜問題点＞
# - クラスで実装した関数を使うには複数段階のステップを踏む必要がある
#   --- 気楽に使うことができない


# インスタンス生成
# --- 変数格納
# --- 関数呼び出し
x = Variable(np.array(0.5))
f = Square()

# 関数実行
y = f(x)

# 確認
print(y.data)

# 改良のヒント
# --- 以下の書き方も可能
# --- 普段の書き方としては不格好（この書き方を関数として実装して内部的に行う）
y2 = Square()(x)
print(y2.data)


# 3 関数と同様の操作性を持たせる工夫 --------------------------------------

# インスタンス生成
# --- 変数格納
x = Variable(np.array(0.5))

# 関数実行
y = square(x)

# 確認
print(y.data)


# 4 バックプロパゲーションの実行 -------------------------------------------

# インスタンス生成
# --- 数値格納
x = Variable(np.array(0.5))

# インスタンス生成
# --- 関数の箱を生成するのではなく、引数を格納して関数的に扱っている
a = square(x)
b = exp(a)
y = square(b)

# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)


# 5 バックプロパゲーションの関数を連続して実行 ----------------------------------

# インスタンス生成
# --- 数値格納
x = Variable(np.array(0.5))

# 関数を連続して適用
y = square(exp(square(x)))

# 逆伝播
y.grad = np.array(1)
y.backward()
print(x.grad)
