---
title: 有限差分法(FDA)とcosine_similarity
description: lgbm のカスタム損失関数を題材に、有限差分法で勾配とヘシアンを近似する話。
toc: true
authors:
  - Horikita Saku
tags:
  - Machine Learning
  - Mathematics
  - LightGBM
categories:
  - Machine Learning
series:
  - Finite Difference
date: '2023-09-11'
lastmod: '2023-09-11'
featuredImage: images/cosine-loss-surface.png
draft: false
weight: 4
---

# はじめに

簡単なものですが、何か書こうと思いました。

[yamaさん](https://twitter.com/rntrnaru)はnishikaのコンペでNegative Cosine Similarity Lossについて言及し、それをlgbmのカスタムロス関数として設定しました。

→ yamaさんの[notebook](https://competition.nishika.com/competitions/finance_ts/topics/615)

この記事は[Qiita](https://qiita.com/HorikitaSaku/items/87cbc00b41c17dbb5b84)にも投稿しています。

この中で有限差分法とその応用はとても興味深い問題だと思いましたので、これを最初のブログにしたい。

ここでは、この方法がこのコンペでどのように機能するかを論じるのではなく、lgbm損失関数の働き方と有限差分法の応用についてお話ししたいと思います。

[有限差分法 Wiki](https://ja.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E5%B7%AE%E5%88%86)

------

# 有限差分法 Finite Difference Analysis

有限差分法（Finite Difference Analysis）は、数値解析や数学的モデリングにおいて一般的に使用される手法の一つです。この手法は、関数や方程式の微分を近似的に計算するために使われます。

![等間隔の格子点 x0..x6 を取った y=f(x)](/images/fda-grid.png)

有限差分法は、前進差分(Forward Difference)、中心差分(Central Difference)、後退差分(Backward Difference)に分けられます。

## 1. **前進差分（Forward Difference）**

前進差分は、ある点 $x$ での関数 $f(x)$ の微分を近似するために、その点から少しだけ前進した点 $x+h$ の関数値と $x$ での関数値の差を用いる手法です。前進差分の公式は次のように表されます。

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h}
$$

ここで、$h$ は非常に小さな値です。前進差分は、$x$ から前に進んだ点での変化率を評価する方法です。

## 2. **中心差分（Central Difference）**

中心差分は、$x$ での関数 $f(x)$ の微分を近似するために、$x$ から前後に進んだ点 $x+h$ と $x-h$ での関数値を用いる手法です。中心差分の公式は次のように表されます。

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

中心差分は、$x$ での変化率をより正確に評価する方法で、大体は、前進差分よりも精密な結果を提供します。

## 3. **後退差分（Backward Difference）**

後退差分は、$x$ での関数 $f(x)$ の微分を近似するために、その点から少しだけ後退した点 $x-h$ での関数値と $x$ での関数値の差を用いる手法です。後退差分の公式は次のように表されます。

$$
f'(x) \approx \frac{f(x) - f(x - h)}{h}
$$

後退差分は、$x$ から後ろに進んだ点での変化率を評価する方法です。

## 4. 違いは何?

この3つの方法は、本質的には関数の値の近くにある点を利用しているわけですが、その違いは前後の違いであって、前後を決めるのは関数自身です。前方の情報が大事なのか、後方の情報が大事なのかです。

一方、一般的には中心差分が最も精度が高いとされています。

## 5. **具体例**

例えば、関数 $f(x)=x^2$ の導関数を $x = 4$ で計算する場合、中心差分法を使うと次のように計算できます。

$$
f'(4) \approx \frac{f(4 + h) - f(4 - h)}{2h}
$$

$h$ を小さくして計算を行います。

------

# 簡単な応用と公式の導出

### ——有限差分法を使って関数の1階と2階の微分をどうやって得るか？

## 1. １階微分——勾配

一階の微分は関数の傾きを表します。

関数 $f(x)$ の点 $x_0$ における一階微分を計算するとします。テイラー級数展開から、$f(x)$ の点 $x_0$ の近くの値を次のように表すことができます：

$$
f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots
$$

このあとは高次の無限小ですから最初の2つだけを残すと次のようになります：

$$
f(x) \approx f(x_0) + f'(x_0)(x-x_0)
$$

上式の $x$ を $x_0 + dx$ と $x_0 - dx$ と取ると、次のようになります：

$$
f(x_0 + dx) \approx f(x_0) + f'(x_0)dx
$$

$$
f(x_0 - dx) \approx f(x_0) - f'(x_0)dx
$$

上の2つの式を減算すると、次のようになります：

$$
f(x_0 + dx) - f(x_0 - dx) \approx 2f'(x_0)dx
$$

したがって、以下の有限差分式が得られます：

$$
f'(x_0) \approx \frac{f(x_0 + dx) - f(x_0 - dx)}{2dx}
$$

## 2. 2階微分——ヘシアン行列（Hessian matrix）

二階微分は一階微分の変化率を表し、曲率とも呼ばれます。一階微分の微分を計算すること、つまり勾配の勾配を計算することです。これはヘシアン行列と呼ばれます。

2階の微分も同様です。しかし、テイラー級数展開は最初の3つだけを残します：

$$
f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2
$$

上式の $x$ を $x_0 + dx$ と $x_0 - dx$ と取ると：

$$
f(x_0 + dx) \approx f(x_0) + f'(x_0)dx + \frac{f''(x_0)}{2!}dx^2
$$

$$
f(x_0 - dx) \approx f(x_0) - f'(x_0)dx + \frac{f''(x_0)}{2!}dx^2
$$

上の2つの式を足すと、次のようになります：

$$
f(x_0 + dx) + f(x_0 - dx) \approx 2f(x_0) + f''(x_0)dx^2
$$

したがって、以下の有限差分式が得られます：

$$
f''(x_0) \approx \frac{f(x_0 + dx) + f(x_0 - dx) - 2f(x_0)}{dx^2}
$$

## 3. cosine_similarityを例にとります

ここではyamaさんのコードにおけるcosine_similarityを例にとります。

2つのベクトル、predとtrueだとします。predとtrue の間の類似性を計算する指標で、次のように表現できます：

$$
\text{Cosine Similarity}(\mathbf{pred}, \mathbf{true}) = \frac{\mathbf{pred} \cdot \mathbf{true}}{\|\mathbf{pred}\| \|\mathbf{true}\|}
$$

$1 - \cos$ を損失として見ると、true を固定したときの損失は次のような形になります。損失は pred が true と同じ向きを指すとき最小になります。

![1-cos 損失の曲面と等高線](/images/cosine-loss-surface.png)

#### １階微分 / 勾配

関数 $f(pred, true)$ については、中心差分を用いて計算します、以下の式が得られます：

$$
\nabla f(\mathbf{pred}, \mathbf{true}) \approx \frac{f(\mathbf{pred} + \Delta\mathbf{pred}, \mathbf{true} + \Delta\mathbf{true}) - f(\mathbf{pred} - \Delta\mathbf{pred}, \mathbf{true} - \Delta\mathbf{true})}{2\epsilon}
$$

ここで、$\Delta\mathbf{pred}$ と $\Delta\mathbf{true}$ は小さなベクトル増加で、$\epsilon$ は小さな正の数です。

#### 2階微分 / ヘシアン行列

$$
\nabla^2 f(\mathbf{pred}, \mathbf{true}) \approx \frac{\nabla f(\mathbf{pred} + \Delta\mathbf{pred}, \mathbf{true} + \Delta\mathbf{true}) - \nabla f(\mathbf{pred} - \Delta\mathbf{pred}, \mathbf{true} - \Delta\mathbf{true})}{2\epsilon}
$$

#### もっとわかりやすく言えば：

ベクトルに対する微分や差分計算では、各要素に対して個別に偏微分や差分を計算します。

これは、ベクトル内の各要素が互いに独立していると仮定されるためです。

例えば、2次元ベクトル $pred = [P_0,P_1]$ の場合、cosine_similarityの１階偏微分（勾配）は以下のように計算されます：

$$
\frac{\partial f}{\partial P_0} \approx \frac{f([P_0 + \Delta P_0, P_1], \mathbf{true}) - f([P_0 - \Delta P_0, P_1], \mathbf{true})}{2\epsilon}
$$

$$
\frac{\partial f}{\partial P_1} \approx \frac{f([P_0, P_1 + \Delta P_1], \mathbf{true}) - f([P_0, P_1 - \Delta P_1], \mathbf{true})}{2\epsilon}
$$

こうして勾配が得られます：

$$
\nabla f(\mathbf{pred},\mathbf{true}) = \left[\frac{\partial f}{\partial P_0}, \frac{\partial f}{\partial P_1}\right]
$$

同様にヘシアン行列が得られます：

$$
\frac{\partial^2 f}{\partial P_0^2} \approx \frac{f([P_0 + 2\Delta P_0, P_1], \mathbf{true}) - 2f([P_0 + \Delta P_0, P_1], \mathbf{true}) + f([P_0, P_1], \mathbf{true})}{\epsilon^2}
$$

$$
\frac{\partial^2 f}{\partial P_1^2} \approx \frac{f([P_0, P_1 + 2\Delta P_1], \mathbf{true}) - 2f([P_0, P_1 + \Delta P_1], \mathbf{true}) + f([P_0, P_1], \mathbf{true})}{\epsilon^2}
$$

$$
\frac{\partial^2 f}{\partial P_0 \partial P_1} = \frac{\partial^2 f}{\partial P_1 \partial P_0} \approx \frac{f([P_0 + \Delta P_0, P_1 + \Delta P_1], \mathbf{true}) - f([P_0 + \Delta P_0, P_1 - \Delta P_1], \mathbf{true})}{4\epsilon^2}
$$

$$
\nabla^2 f(\mathbf{pred}, \mathbf{true}) = \left[\frac{\partial^2 f}{\partial P_0^2}, \frac{\partial^2 f}{\partial P_1^2}, \frac{\partial^2 f}{\partial P_0 \partial P_1}\right]
$$

------

# Python

yamaさんのコードをそのまま引用させていただきます。

若干の修正を加えましたが、最後に説明します。

```python
def derivative(y_pred, y_true, dx):
    n = y_pred.shape[0]
    grad = np.zeros(n)
    hess = np.zeros(n)
    z_true = np.sqrt((y_true ** 2).sum())
    f_0 = 1 -1 * (y_pred * y_true).sum() / np.sqrt((y_pred ** 2).sum()) / z_true
    for i in range(n):
        y_pred[i] = y_pred[i] - dx
        f_m = 1 -1 * (y_pred * y_true).sum() / np.sqrt((y_pred ** 2).sum()) / z_true
        y_pred[i] = y_pred[i] + dx + dx
        f_p = 1 -1 * (y_pred * y_true).sum() / np.sqrt((y_pred ** 2).sum()) / z_true
        y_pred[i] = y_pred[i] - dx
        grad[i] = (f_p - f_m) / (dx * 2)
        hess[i] = (f_p + f_m - 2 * f_0) / (dx ** 2)

    return grad, hess

# 負のコサイン類似度
# LightGBMの目的関数に指定する。
def negative_cosine_similarity_loss(y_pred, data):
    y_true = np.double(data.get_label())
    grad, hess = derivative(y_pred, y_true, 1e-3)

    return grad, hess
```

このうち、f0は元の値の結果：

$$
f_0 = 1 - \frac{\sum{(y_{pred} \cdot y_{true})}}{\|y_{pred}\| \cdot \|y_{true}\|}
$$

つまり $(1 - cosine\ similarity)$ 類似度です。1で引く目的は値域を $[-1,1]$ から $[2,0]$ に変換することです。変換するとangular_lossになります。

このようなメリットは、損失関数の値を下げる方法については、汎用性があるということです。ただ、悪い点は、実験時の損失値が直接的ではないことです。

状況に応じて値域を変えるかどうかを判断する必要があります。

$fm\ (f_{minus})$: $y_{pred}[i]$ から微小な正の数 $dx$ を差し引いた関数値です。次の数式で表すことができます。

$$
fm = f(y_{pred} - [0, ..., 0, dx, 0, ..., 0], y_{true})
$$

$fp\ (f_{plus})$: $y_{pred}[i]$ に微小な正の数 $dx$ を加えた関数値です。次の数式で表すことができます。

$$
fp = f(y_{pred} + [0, ..., 0, dx, 0, ..., 0], y_{true})
$$

簡単に言うと、予測値を繰り返し、それぞれの予測値に有限差分法を1回使って、grad, hess、つまり勾配とヘシアン行列をつくります。

するとお気づきかもしれませんが、ヘシアン行列には先ほど述べた3つ目の成分はありません。混合偏微分 $\frac{\partial^2 f}{\partial P_0 \partial P_1}$ はありません。

これはLightGBMがニュートン法のような最適化されたアルゴリズムではなく、基礎学習器としてGradient Boosting Treesを使っているからです。

lgbmはヘシアン行列の3番目の成分を計算して使用することなく、主に勾配情報を用いて構築します。この場合、１階微分(勾配)と2階微分(ヘシアン行列の対角線要素 e.g. $[\frac{\partial^2 f}{\partial P_0^2}, \frac{\partial^2 f}{\partial P_1^2}]$)だけを計算してモデルを作ればよいのです。

これは、ヘシアン行列の非対角線要素を考慮することなく、tree model を構築するための分割規則とleaf値の更新に使われます。これにより、計算コストが削減され、より効率的なモデルトレーニングが可能になります。
