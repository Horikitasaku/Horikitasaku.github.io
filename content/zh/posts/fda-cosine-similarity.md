---
title: 有限差分法（FDA）与 cosine_similarity
description: 以一个 lgbm 自定义损失为例，看有限差分如何近似梯度和 Hessian。
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

# 写在前面

没什么复杂的东西，就是想写点什么。

[yama](https://twitter.com/rntrnaru) 在 nishika 的一场比赛里提到了 Negative Cosine Similarity Loss，并把它做成了 lgbm 的自定义损失函数。

→ yama 的 [notebook](https://competition.nishika.com/competitions/finance_ts/topics/615)

这篇文章同时也发在 [Qiita](https://qiita.com/HorikitaSaku/items/87cbc00b41c17dbb5b84) 上。

我觉得里面有限差分法及其用法是个很有意思的点，所以就拿它当第一篇博客的题材了。

这里我不打算讨论这个损失在那场比赛里效果如何，只想聊聊 lgbm 损失函数是怎么工作的，以及有限差分法是怎么用上的。

[有限差分法 Wiki](https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E5%B7%AE%E5%88%86%E6%B3%95)

------

# 有限差分法 Finite Difference Analysis

有限差分法（Finite Difference Analysis）是数值分析和数学建模里很常用的一类方法，用来近似计算函数或方程的导数。

![在等间隔网格点 x0..x6 上采样的 y=f(x)](images/fda-grid.png)

它分为前向差分（Forward Difference）、中心差分（Central Difference）和后向差分（Backward Difference）三种。

## 1. **前向差分（Forward Difference）**

要近似 $f(x)$ 在某点 $x$ 处的导数，前向差分用稍微往前一点的点 $x+h$ 处的函数值，减去 $x$ 处的函数值。公式如下：

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h}
$$

其中 $h$ 是一个很小的数。换句话说，它衡量的是从 $x$ 往前看的变化率。

## 2. **中心差分（Central Difference）**

要近似 $f(x)$ 在 $x$ 处的导数，中心差分同时用两侧的点 $x+h$ 和 $x-h$ 处的函数值。公式如下：

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

中心差分对 $x$ 处变化率的估计更准，一般来说比前向差分更精确。

## 3. **后向差分（Backward Difference）**

要近似 $f(x)$ 在 $x$ 处的导数，后向差分用稍微往后一点的点 $x-h$ 处的函数值，从 $x$ 处的函数值里减去它。公式如下：

$$
f'(x) \approx \frac{f(x) - f(x - h)}{h}
$$

它衡量的是从 $x$ 往后看的变化率。

## 4. 区别在哪？

这三种方法本质上都是利用函数取值附近的点，区别只在于往前还是往后——而往哪边走，是由函数本身决定的：到底是前面的信息更重要，还是后面的信息更重要。

不过一般认为，三者里中心差分的精度最高。

## 5. **一个具体例子**

比如要算 $f(x)=x^2$ 在 $x = 4$ 处的导数，用中心差分：

$$
f'(4) \approx \frac{f(4 + h) - f(4 - h)}{2h}
$$

把 $h$ 取得越来越小再算就行。

------

# 简单应用与公式推导

### ——怎么用有限差分法得到函数的一阶和二阶导数？

## 1. 一阶导数——梯度

一阶导数表示函数的斜率。

设我们要算 $f(x)$ 在点 $x_0$ 处的一阶导数。由泰勒展开，$f(x)$ 在 $x_0$ 附近的取值可以写成：

$$
f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots
$$

后面都是高阶无穷小，只保留前两项就得到：

$$
f(x) \approx f(x_0) + f'(x_0)(x-x_0)
$$

把上式里的 $x$ 取成 $x_0 + dx$ 和 $x_0 - dx$：

$$
f(x_0 + dx) \approx f(x_0) + f'(x_0)dx
$$

$$
f(x_0 - dx) \approx f(x_0) - f'(x_0)dx
$$

两式相减：

$$
f(x_0 + dx) - f(x_0 - dx) \approx 2f'(x_0)dx
$$

于是得到有限差分式：

$$
f'(x_0) \approx \frac{f(x_0 + dx) - f(x_0 - dx)}{2dx}
$$

## 2. 二阶导数——Hessian 矩阵

二阶导数表示一阶导数的变化率，也叫曲率。它是导数的导数，也就是梯度的梯度，这正是 Hessian 矩阵刻画的东西。

二阶的做法一样，只不过泰勒展开要保留前三项：

$$
f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2
$$

把 $x$ 取成 $x_0 + dx$ 和 $x_0 - dx$：

$$
f(x_0 + dx) \approx f(x_0) + f'(x_0)dx + \frac{f''(x_0)}{2!}dx^2
$$

$$
f(x_0 - dx) \approx f(x_0) - f'(x_0)dx + \frac{f''(x_0)}{2!}dx^2
$$

两式相加：

$$
f(x_0 + dx) + f(x_0 - dx) \approx 2f(x_0) + f''(x_0)dx^2
$$

于是得到有限差分式：

$$
f''(x_0) \approx \frac{f(x_0 + dx) + f(x_0 - dx) - 2f(x_0)}{dx^2}
$$

## 3. 以 cosine_similarity 为例

这里拿 yama 代码里的 cosine_similarity 来举例。

设有两个向量 pred 和 true，余弦相似度衡量它们之间的相似程度：

$$
\text{Cosine Similarity}(\mathbf{pred}, \mathbf{true}) = \frac{\mathbf{pred} \cdot \mathbf{true}}{\|\mathbf{pred}\| \|\mathbf{true}\|}
$$

把 $1 - \cos$ 当作损失、固定 true 后，损失的形状如下——只要 pred 和 true 方向一致，损失就取到最小。

![1-cos 损失的曲面与等高线](images/cosine-loss-surface.png)

#### 一阶导数 / 梯度

对 $f(pred, true)$ 用中心差分计算，得到：

$$
\nabla f(\mathbf{pred}, \mathbf{true}) \approx \frac{f(\mathbf{pred} + \Delta\mathbf{pred}, \mathbf{true} + \Delta\mathbf{true}) - f(\mathbf{pred} - \Delta\mathbf{pred}, \mathbf{true} - \Delta\mathbf{true})}{2\epsilon}
$$

其中 $\Delta\mathbf{pred}$ 和 $\Delta\mathbf{true}$ 是很小的向量增量，$\epsilon$ 是一个很小的正数。

#### 二阶导数 / Hessian 矩阵

$$
\nabla^2 f(\mathbf{pred}, \mathbf{true}) \approx \frac{\nabla f(\mathbf{pred} + \Delta\mathbf{pred}, \mathbf{true} + \Delta\mathbf{true}) - \nabla f(\mathbf{pred} - \Delta\mathbf{pred}, \mathbf{true} - \Delta\mathbf{true})}{2\epsilon}
$$

#### 说得再直白一点：

对向量做求导或差分时，是对每个分量分别去算偏导或差分，一次一个。

这是因为我们假设向量里的各个分量彼此独立。

比如对二维向量 $pred = [P_0,P_1]$，cosine_similarity 的一阶偏导（梯度）这样算：

$$
\frac{\partial f}{\partial P_0} \approx \frac{f([P_0 + \Delta P_0, P_1], \mathbf{true}) - f([P_0 - \Delta P_0, P_1], \mathbf{true})}{2\epsilon}
$$

$$
\frac{\partial f}{\partial P_1} \approx \frac{f([P_0, P_1 + \Delta P_1], \mathbf{true}) - f([P_0, P_1 - \Delta P_1], \mathbf{true})}{2\epsilon}
$$

于是得到梯度：

$$
\nabla f(\mathbf{pred},\mathbf{true}) = \left[\frac{\partial f}{\partial P_0}, \frac{\partial f}{\partial P_1}\right]
$$

Hessian 矩阵同理：

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

下面直接引用 yama 的代码。

我做了一点小改动，最后会说明。

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

# 负余弦相似度
# 作为 LightGBM 的目标函数传入。
def negative_cosine_similarity_loss(y_pred, data):
    y_true = np.double(data.get_label())
    grad, hess = derivative(y_pred, y_true, 1e-3)

    return grad, hess
```

其中 f0 是原始点处的取值：

$$
f_0 = 1 - \frac{\sum{(y_{pred} \cdot y_{true})}}{\|y_{pred}\| \cdot \|y_{true}\|}
$$

也就是 $(1 - cosine\ similarity)$。用 1 去减，是为了把取值范围从 $[-1,1]$ 映射到 $[2,0]$；变换之后就成了 angular_loss。

这样做的好处是通用——不管你用什么办法去把损失值往下压，都能直接套用。坏处是实验时看到的损失值不那么直观。

要不要做这个范围变换，得看具体情况来定。

$fm\ (f_{minus})$：把 $y_{pred}[i]$ 减去一个很小的正数 $dx$ 之后的函数值，可以写成：

$$
fm = f(y_{pred} - [0, ..., 0, dx, 0, ..., 0], y_{true})
$$

$fp\ (f_{plus})$：把 $y_{pred}[i]$ 加上一个很小的正数 $dx$ 之后的函数值：

$$
fp = f(y_{pred} + [0, ..., 0, dx, 0, ..., 0], y_{true})
$$

简单说，就是遍历每个预测值，对每一个用一次有限差分法，攒出 grad 和 hess，也就是梯度和 Hessian 矩阵。

到这里你可能已经注意到了：这里的 Hessian 矩阵里没有前面提到的第三个分量，也就是混合偏导 $\frac{\partial^2 f}{\partial P_0 \partial P_1}$。

这是因为 LightGBM 不是牛顿法那类优化算法，它用的基学习器是 Gradient Boosting Trees。

lgbm 主要靠梯度信息来构建模型，从头到尾都不会去算、也不会用到 Hessian 矩阵的第三个分量。所以这种情况下，只要算出一阶导数（梯度）和二阶导数中 Hessian 矩阵的对角线元素（比如 $[\frac{\partial^2 f}{\partial P_0^2}, \frac{\partial^2 f}{\partial P_1^2}]$）就足够建模了。

这些量被用来决定构建树模型时的分裂规则和叶子值更新，完全不需要考虑 Hessian 矩阵的非对角元素。这样既降低了计算开销，训练也更高效。