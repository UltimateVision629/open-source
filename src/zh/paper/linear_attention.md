---
title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
date: 2025-12-13
category:
  - paper
tag:
  - paper
---

## 动机
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

在transformer的自注意力模块当中。如果句子的长度为$n$, 词向量的维度为$d_k$ , 先将 $Q(n×d_k)$ 与 $K^\top(d_k×n)$ 相乘, 则时间复杂度为 $\mathcal{O}(d_kn^2)$。 $QK^\top(n×n)$ 经过 $Softmax$ 和 缩放 后的矩阵与 $V(n×d_k)$ 相乘的时间复杂度同样为 $\mathcal{O}(d_kn^2)$。当$n >> d_k$时，时间复杂度为句子长度的平方，这就导致了注意力模块中的二次时间复杂度问题，当上下文过长时，时间复杂度较高。\
\
如果我们先将 $K^\top(d_k×n)$ 与 $V(n×d_k)$ 相乘，则时间复杂度为$\mathcal{O}(d_k*n*d_k) = \mathcal{O}(d_k^2*n)$。 同样，再将 $Q(n×d_k)$  与 $K^TV(d_k×d_k)$，时间同样为$\mathcal{O}(n*d_k*d_k) = \mathcal{O}(d_k^2*n)$。当上下文长度较大时，$n >> d_k$, 我们可以认为时间复杂度为线性时间复杂度。但由于 $Softmax$ 的限制，我们无法做到先计算 $K^TV$。

## 自注意力机制的理解
![点积表示两个向量的相似度](./image/linear_attention/dot.png)
数学上我们知道，两个向量的**点积**表示两个向量在空间中的**接近程度**。对于两个长度为1的空间向量，点积结果越接近1，表示两个向量越相似。在NLP当中，两个词向量的点积可以反映两个词向量的相关程度。
![注意力矩阵的含义](./image/linear_attention/QKdetail.png)
基于以上数学上的意义，我们可以将自注意力中的 $Q·K^\top$ 理解为当前词向量与句子中的其他词向量的相关程度。具体的，$Q·K^\top$ 结果矩阵中的第一行第二列，可以看作是 $Q$ 矩阵中的第一行和 $K^\top$ 矩阵中的第二列做点积，即表示在该句子中 *小明* 和 *最* 的相关程度。示意图如上图所示。 

## 特征映射与核函数

本节给出特征映射与核函数定义，帮助读者更好的理解后面的内容。\
\
![二维升三维后线性可分](./image/linear_attention/SVM.png)
#### 什么是**特征映射**（Feature Mapping）？
在SVM中，由于原始数据点所在的平面可能不足以使用一个线性决策边界将所有点分类。有时候我们需要使用**特征映射**将数据点映射到高维空间使得数据点在新的高维特征空间中变得**线性可分**。\
如上图所示，原先数据点在二维空间线性不可分，将所有点映射至三维空间后，数据点变的线性可分。\
\
**特征映射**（Feature Mapping），通常用 $\Phi(\mathbf{x})$ 或 $\phi(\mathbf{x})$ 表示，是一种将原始输入空间 $\mathcal{X}$ 中的数据点 $\mathbf{x}$ 转换（映射）到一个新的、更高维度的特征空间 $\mathcal{H}$ 的函数。
$$\phi: \mathcal{X} \to \mathcal{H}$$

#### 什么是**核函数**（Kernel Function）？
核函数（Kernel Function），通常用 $K(\mathbf{x}, \mathbf{z})$ 表示，是一种直接计算在特征空间 $\mathcal{H}$ 中两个映射后的向量 $\phi(\mathbf{x})$ 和 $\phi(\mathbf{z})$ 之间内积的方法，而无需显式地执行特征映射 $\phi$。根据核函数的定义，它必须满足以下关系：

$$K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x}) \cdot \phi(\mathbf{z})$$
其中，$\cdot$ 表示特征空间 $\mathcal{H}$ 中的内积（点积）。

#### 核心思想：核技巧（Kernel Trick）
核函数的核心价值在于实现了著名的**核技巧**：
* 许多机器学习算法（如SVM）只需要计算数据点之间的内积（相似度）。如果要显式地进行高维特征映射 $\phi(\mathbf{x})$，计算量往往非常巨大（比如映射到无限维）。
* 核函数允许我们在原始的低维空间 $\mathcal{X}$ 中计算 $K(\mathbf{x}, \mathbf{z})$，这个计算结果等价于高维特征空间 $\mathcal{H}$ 中的内积 $\phi(\mathbf{x}) \cdot \phi(\mathbf{z})$，但计算效率要高得多。

## 自注意力的核函数理解
[Yao-Hung Hubert Tsai](https://arxiv.org/abs/1908.11775) 等人的论文中指出。transformer中的自注意力可以认为是其核函数表达中的一种特殊形式。\
\
从本文**自注意力机制的理解**章节中，我们得知，计算 $Q·K^\top$ 本质上是在计算向量之间的相关性。[Yao-Hung Hubert Tsai](https://arxiv.org/abs/1908.11775) 等人将这种相似度计算与核函数联系起来。提出自注意力只是其核函数表达中的一种特殊形式。\
其核函数表达公式为

$$
A_i = \frac{\sum_{j=1}^{N} \text{K} (Q_i, K_j) V_j}{\sum_{j=1}^{N} \text{K} (Q_i, K_j)}
$$
\
其中 $A_i$ 表示注意力结果矩阵的第 $i$ 行，$Q_i$ 表示query矩阵的第 i 行，$K_j$ 表示 key 矩阵第 j 行，$V_j$ 表示 value 矩阵第 j 行。(公式可能与原论文不同,此处为方便理解)\
\
在 transformer 中, $K(q, k) = \exp\left( \frac{q^\top k}{\sqrt{D}} \right)$。

## 利用特征映射转化为线性复杂度

如果 $K(Q_i, K_j)$ 可以分解成两个特征映射(根据核函数的定义)。

则
$$
A_i = \frac{\sum_{j=1}^N \phi(Q_i) \phi(K_j)^\top V_j}{\sum_{j=1}^N \phi(Q_i) \phi(K_j)^\top}

$$
$\phi(Q_i)$ 在求和符号内与 j 无关，可以提取出来。即：

$$
A_i = \frac{\phi(Q_i) \sum_{j=1}^N \phi(K_j)^\top V_j}{\phi(Q_i) \sum_{j=1}^N \phi(K_j)^\top} \
$$
在上面这个式子中，对于每个 $i$ ，$\sum_{j=1}^N \phi(K_j)^\top V_j$ 可以提前算出来。最高时间复杂度即为$\sum_{j=1}^N \phi(K_j)^\top V_j$ 这个式子。假设特征维度为 $d_k$, $\phi(K_j)^\top V_j$的时间复杂度为 $\mathcal{O}({d_k}^2)$, 再累加 $N$ 次,时间复杂度为 $\mathcal{O}(N{d_k}^2)$。当 $N >> d_k$ 时，时间复杂度为线性。\
\
通过以上推导可知，如果我们能将核函数拆解为两个特征函数，则我们就能将transformer中的注意力计算时间复杂度降为线性。可惜的是，指数核对应的特征函数是无限维的，我们无法在有限时间复杂度内进行特征映射。
\
\
作者在文中采用以下特征函数，并认为其性能与transformer持平。
$$\phi (x) = \text{elu}(x) + 1 \quad$$


## 如何调试

下载项目并以可编辑模式安装当前目录下的Python包。
~~~bash
git clone https://github.com/idiap/fast-transformers.git
cd fast-transformers
pip install -e .
~~~

配置 `.vscode/launch.json`
~~~json
{
    "configurations": [
        {
        "name": "Python: Test current file",
        "type": "python",
        "request": "launch",
        "module":"pytest",
        "cwd":"${workspaceFolder}",
        "args": [
        "-v",
        "-s",
        "${file}"
        ],
        "console": "integratedTerminal",
        "env": {
                "BENCHMARK_TESTS": "1"
            }
        }
    ]
}
~~~

使用vscode打开任意test文件按`F5`即可进行单步调试。

**参考文献**\
[1]  https://arxiv.org/abs/2006.16236 \
[2]  https://zhuanlan.zhihu.com/p/719570478 \
[3]  https://arxiv.org/abs/1908.11775

