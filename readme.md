## Spatio-temporal modeling 论文列表（主要是graph convolution相关）

应广大小伙伴建议，建了一个GCN交流群，感兴趣的小伙伴可以加群哇，比心～

<div align=center>
 <img src="Wechat.jpeg" width = "40%" height = "40%" alt="图片名称" align=center/>
</div>

小白一枚，接下来希望在时空建模上有点见解，图是数据表示非常自然的方式，现在在处理图上数据的任务时常用network embedding的方法和的geometric model方法。network embedding初衷是把图上数据表示成张量形式，可以满足deep learning model的输入，但这种方法在学数据表示时存在信息压缩；后者则是修改模型，使得满足输入结构化，即输入数据保留图的形式。接下来打算在第二种思想上展开，保留图上数据的空间约束，同时有些图上数据是时间序列的，如路网上每时刻节点的流量速度等，如何对时间序列上的网络数据用deep model建模，保留空间约束是接下来的学习方向。
#### 这里会整理些近期看的论文及简单描述(可能不准确)，会持续更新，希望同样研究这个方向的小伙伴可以一起交流～

### 综述篇
- [Geometric deep learning: going beyond Euclidean data (IEEE Signal Processing Magazine 2017)](https://arxiv.org/pdf/1611.08097.pdf)
> 关于非欧数据（主要就是图数据和流形数据）的深度学习的review

- [Graph Neural Networks:A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)
> 关于GCN领域的方法和应用的综述

### 方法篇
- [The graph neural network model （TNN 2009）](https://repository.hkbu.edu.hk/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1000&context=vprd_ja)

- [The emerging field of signal processing on graphs (IEEE Signal Processing Magazine 2013)](https://arxiv.org/pdf/1211.0053.pdf)
> 解释怎么处理图上的信号，通过图上拉普拉斯矩阵特征值的smooth性质类比傅立叶系数，利用特征向量做图上傅立叶变换，解释如何做图上的卷积，滤波运算。

- [spectral networks and deep locally connected networks on graphs (ICLR2014)](https://arxiv.org/pdf/1312.6203.pdf)
> 把图上的CNN extend 到graph上，提出空间和spectral两种方法，空间上每层有几个cluster，学习相邻层节点之间的权重。Spectral用上文中拉普拉斯矩阵的特征向量变换到谱域，通过谱域的点乘再做傅立叶逆变换得到卷积的结果，学习参数O(n)个，没有显示的localize，且计算量较大。

- (GWNN)[Graph Wavelet Neural Network (ICLR2019)](https://openreview.net/forum?id=H1ewdiR5tQ)
> 用小波变换代替傅立叶变换实现图卷积。小波变换相对于傅立叶变换具有局部性，稀疏性和可解释性的性质，这些使得应用小波变换的图卷积神经网络更加高效，也满足了localize性质。

- [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/abs/0912.3848)
> 图上小波变换的一些理论知识，包括如何加速计算，licalize的性质保证等。

- [deep convolutional networks on graph-structured data (ICLR2015)](https://arxiv.org/pdf/1506.05163.pdf)      
   [PPT](http://web.eng.tau.ac.il/deep_learn/wp-content/uploads/2017/03/Deep-Convolutional-Networks-on-Graph-Structured-Data.pdf)
> 在上一篇的基础上希望减少参数，类比图像上CNN的局部连接，希望定义的kernel只在节点和它的近邻之间产生联系，定义interpolation kernel作用在要学的参数上。另外这篇文章还提出建图的方法，某些具有相互联系的数据却并不存在自然的图，构图方法有unsupervised graph estimation和supervised graph estimation。

- (Cheby-Net)[convolutional neural networks on graphs with fast localized spectral filtering (NIPS2016)](https://arxiv.org/pdf/1606.09375.pdf)
> 包括两个部分工作，一个就是用多项式kernel，降低参数，并且拉普拉斯矩阵K次方上，两点最短路径大于K，则取值为0 的性质保持localize，并用切比雪夫多项式近似来加速，且省掉拉普拉斯矩阵特征分解的步骤。另外提出pooling的方法，通过添加fake node贪心的将所有节点组织成二叉树，在树上做pooling。

- (GCN)[semi-supervised classification with graph convolutional networks (ICLR2017)](https://arxiv.org/pdf/1609.02907.pdf)
> 对上文的方法作了进一步简化，首先把多项式kernel的项控制在两个，并且约束两个参数值相同，主要是在citation网络的节点分类上，训练时每一类有20个带标签数据，共7个类，预测时结果显著比原来方法好

- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (NIPS2015)](https://arxiv.org/pdf/1506.04214.pdf)
> 利用雷达降雨量预测，不同雷达之间有空间联系，同一个雷达的序列数据存在时间联系，用传统CNN刻画雷达间的联系，用LSTM刻画时间联系，把LSTM中参数和输入数据的矩阵乘法替换成卷积，使得同时建模空间和时间约束，虽然输入数据依旧是标准张量，但是把时间空间结合起来。

- [Structured Sequence Modeling with Graph Convolutional Recurrent Networks (ICONIP 2017)](https://arxiv.org/pdf/1612.07659.pdf)
> 把时间数据和空间数据结合起来，方法有输入数据做完图卷积然后再输入LSTM，或者把LSTM中的矩阵乘法替换成图卷积。

- [Convolutional Networks on Graphs for Learning Molecular Fingerprints (NIPS2015)](https://arxiv.org/pdf/1509.09292.pdf)
> 分子是原子及连边的图，任务是预测分子的属性，把原始方法中的hash函数替换成用一层neural network学一个smooth的function，利用hash结果取余找index的过程替换成softmax。输入是图的形式，但是处理方法和上面论文不太一样。

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (AAAI 2018)](https://arxiv.org/pdf/1801.07455.pdf)
> 使用图卷积处理动作识别。以人体骨骼的关节为图的顶点，以人体的躯干为边，将连续的时间片上相同的关节连接起来，构造三维的时空图结构。通过卷积在图像上的定义，类比出卷积在图上的定义，对顶点的邻居进行子集划分，每个子集与对应的权重相乘，得到时空图卷积的定义。使用Kipf & Welling 2017的公式进行实现。

下面几篇在处理dynamic graph，把传统CNN与图上CNN统一起来

- [Transfer learning for deep learning on graph-structured data (AAAI 2017)](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14803/14387)

- [Graph Convolution: A High-Order and Adaptive Approach](https://arxiv.org/pdf/1706.09916.pdf)
> Kipf & Welling的方法使用的是一阶邻居，作者在本文中利用邻接矩阵的k次幂，提出了k阶邻居的图卷积方法。将k阶邻接矩阵与顶点特征矩阵拼接，与权重矩阵Q进行线性组合，构造出可以同时捕获顶点特征与图结构性质的自适应卷积核。在citation graphs上对顶点分类，以及在分子性质预测上进行了测试。

- [Learning Graph Convolution Filters from Data Manifold](https://arxiv.org/pdf/1710.11577.pdf)

- [Diffusion-Convolutional Neural Networks (NIPS 2016)](https://arxiv.org/pdf/1511.02136.pdf)
> 在卷积操作中融入了h-hop转移概率矩阵，通过对每个顶点计算该顶点到其他所有顶点的转移概率与特征矩阵的乘积，构造顶点新的特征表示，即diffusion-convolutional representation，表征顶点信息的扩散，然后乘以权重矩阵W，加激活函数，得到卷积的定义。在顶点分类和图分类上做了测试。作者提到的模型缺陷是空间复杂度高，以及模型不能捕获尺度较大的空间依赖关系。

- [Dynamic Graph Convolutional Networks](https://arxiv.org/pdf/1704.06199.pdf)

- [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/pdf/1806.03536.pdf)
> 针对不同节点可能邻域范围不同，提出了Jumping Knowledge Network，分别采用了Concat、Max-Pooling、LSTM-Atten作为最后一层的aggregator，最后在Citeseer & Cora和Reddit上做实验验证

下面几篇还没有读

- [Geometric deep learning on graphs and manifolds using mixture model CNNs (CVPR2017)](https://arxiv.org/pdf/1611.08402.pdf)
> graph不具备平移不变性，这篇文章提出了一个统一的框架，给每个节点定义统一个数的weighting function，每个weighting function 把所有周围节点映射成一个表达，使得每个节点的局部结构虽然不同，但是weighting function作用后，个数相同，convolution kernel定义在weighting function上. 并且作者给出GCN 在框架下的解释，并给出mixture gaussian model作为weighting function。

- [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs (CVPR2017)](https://arxiv.org/pdf/1704.02901.pdf)

- [Protein interface prediction using graph convolutional networks (NIPS2017)](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf)

- [Gated Graph Sequence Neural Networks (ICLR2016)](https://arxiv.org/pdf/1511.05493.pdf)

### 应用篇

注：后续有不同研究领域的小伙伴，欢迎继续添加相应领域的paper~

#### 交通预测

- [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (ICLR 2018)](https://arxiv.org/pdf/1707.01926.pdf)
> DCRNN，是[Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659.pdf)中的方法的应用，对其中Defferrard 的 k 阶切比雪夫图卷积进行了替换，使用了[Teng et al., 2016](http://www-bcf.usc.edu/~shanghua/teaching/Fall2016-670/networkDataAnalysisPrintedBook.pdf)的图上随机游走的平稳分布的闭式解，定义了扩散卷积(Diffusion convolution)，其实是一种有向图卷积，使用前一篇论文中的模型2，用 GRU 构造了 DCRNN(DCGRU)，对道路传感器网络上下一时刻的速度预测，取得了state of the art的表现，12个点预测12个点，在METR-LA 和 PEMS 上进行了实验，数据已公开。

- [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting (IJCAI 2018)](https://arxiv.org/pdf/1709.04875.pdf)
> STGCN，分别采用 ChebyNet 和 GCN 两种方式，将图卷积网络应用在交通流短时预测上，图卷积做空间关系建模，一维卷积做时间关系建模，交替迭代地组成时空卷积块，堆叠两个块构成模型，最终在 PEMS 和北京市两个数据集上进行实验验证，12个点预测12个点。

- [Multistep Speed Prediction on Traffic Networks: A Graph Convolutional Sequence-to-Sequence Learning Approach with Attention Mechanism（TRC 2019）](https://arxiv.org/ftp/arxiv/papers/1810/1810.10237.pdf)
> 清华大学和高德地图合作的一项研究。作者采用了 GCN + Seq2Seq + Attention 的混合模型，将路网中的边构建成图中的结点，在 GCN 上做了改进，将邻接矩阵扩展到 k 阶并与一个权重矩阵相乘，类似 HA-GCN(2016)，实现了邻居信息聚合时权重的自由调整，可以处理有向图。时间关系上使用 Seq2Seq + Attention 建模，完成了北京市二环线的多步的车速预测，对比的方法中没有近几年出现的时空预测模型。

- [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting （AAAI 2019）](https://github.com/Davidham3/ASTGCN)
> ASTGCN，使用三个组件对时间序列上的近期、日周期、周周期三个模式进行建模，每个组件使用 K 阶切比雪夫图卷积捕获空间关系，使用一维卷积捕获时间关系，在 K 阶切比雪夫多项式展开的图卷积内融入了注意力机制来捕获空间动态性，在时间维上使用注意力机制让模型动态地捕获时间关系。在 PeMS 数据集上进行了实验，数据已公开。

- [Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting (AAAI 2019)](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf)
> ST-MGCN，网约车需求预测，T个点预测1个点。空间依赖建模上：以图的形式表示数据，从空间地理关系、区域功能相似度、区域交通连通性三个角度构造了三个不同的图，提出了多图卷积，分别用 k 阶 ChebNet 对每个图做图卷积，然后将多个图的卷积结果进行聚合(sum, average 等)成一个图；时间依赖建模上：提出了融合背景信息的 Contextual Gated RNN (CGRNN)，用 ChebNet 对每个结点卷积后，得到他们的邻居表示，即每个结点的背景信息表示，与原结点特征拼接，用一个两层全连接神经网络计算出 T 个权重，将权重乘到历史 T 个时刻的图上，对历史值进行缩放，然后用一个共享的 RNN，针对每个结点形成的长度为 T 的时间序列建模，得到每个结点新的时间表示。最后预测每个点的网约车需求，对比的深度学习方法有上述的 DCRNN 和 STGCN 两个，数据是北京和上海的网约车需求数据。
