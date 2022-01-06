# ******************************************************************************
# Title       : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage       : 1 微分を自動で求める
# Step        : 05 バックプロパゲーションの理論
# Create Date : 2021/10/07
# Update Date : 2022/01/06
# Page        : P25 - P29
# ******************************************************************************


# ＜概要＞
# - バックプロパゲーションは数値微分でも課題となった｢計算精度｣と｢計算コスト｣の課題を解消する
# - 連結した複数の関数(合成関数)の微分を、構成する各関数の積に分解することを連鎖律(チェインルール)という
