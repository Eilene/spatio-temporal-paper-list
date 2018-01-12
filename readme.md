## Spatio-temporal modeling 论文列表（主要是graph convolution相关）
小白一枚，接下来希望在时空建模上有点见解，图是数据表示非常自然的方式，现在在处理图上数据的任务时常用network embedding的方法和的grometric model方法。network embedding初衷是把图上数据表示成张量形式，可以满足deep learning model的输入，但这种方法在学数据表示时存在信息压缩；后者则是修改模型，使得满足输入结构化，即输入数据保留图的形式。接下来打算在第二种思想上展开，保留图上数据的空间约束，同时有些图上数据是时间序列的，如路网上每时刻节点的流量速度等，如何对时间序列上的网络数据用deep model建模，保留空间约束是接下来的学习方向。
#### 这里会整理些近期看的论文及简单描述(可能不准确)，会持续更新，希望同样研究这个方向的小伙伴可以一起交流～
- [The graph neural network model （TNN 2009）](https://repository.hkbu.edu.hk/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1000&context=vprd_ja)

- [The emerging field of signal processing on graphs (IEEE Signal Processing Magazine 2013)](https://arxiv.org/pdf/1211.0053.pdf)
> 解释怎么处理图上的信号，通过图上拉普拉斯矩阵特征值的smooth性质类比傅立叶系数，利用特征向量做图上傅立叶变换，解释如何做图上的卷积，滤波运算。

- [spectral networks and deep locally connected networks on graphs (ICLR2014)](https://arxiv.org/pdf/1312.6203.pdf)
> 把图上的CNN extend 到graph上，提出空间和spectral两种方法，空间上每层有几个cluster，学习相邻层节点之间的权重。Spectral用上文中拉普拉斯矩阵的特征向量变换到谱域，通过谱域的点乘再做傅立叶逆变换得到卷积的结果，学习参数O(n)个，没有显示的localize，且计算量较大。

- [deep convolutional networks on graph-structured data (ICLR2015)](https://arxiv.org/pdf/1506.05163.pdf)      
   [PPT](http://web.eng.tau.ac.il/deep_learn/wp-content/uploads/2017/03/Deep-Convolutional-Networks-on-Graph-Structured-Data.pdf)
> 在上一篇的基础上希望减少参数，类比图像上CNN的局部连接，希望定义的kernel只在节点和它的近邻之间产生联系，定义interpolation kernel作用在要学的参数上。另外这篇文章还提出建图的方法，某些具有相互联系的数据却并不存在自然的图，构图方法有unsupervised grapp estimation和supervised graph estimation。

- [convolutional neural networks on graphs with fast localized spectral filtering (NIPS2016)](https://arxiv.org/pdf/1606.09375.pdf)
> 包括两个部分工作，一个就是用多项式kernel，降低参数，并且拉普拉斯矩阵K次方上，两点最短路径大于K，则取值为0 的性质保持localize，并用切比雪夫多项式近似来加速，且省掉拉普拉斯矩阵特征分解的步骤。另外提出pooling的方法，通过添加fake node贪心的将所有节点组织成二叉树，在树上做pOoling。

- [semi-supervised classification with graph convolutional networks (ICLR2017)](https://arxiv.org/pdf/1609.02907.pdf)
> 对上文的方法作了进一步简化，首先把多项式kernel的项控制在两个，并且约束两个参数值相同，主要是在citation网络的节点分类上，训练时每一类有20个带标签数据，共7个类，预测时结果显著比原来方法好

- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (NIPS2015)](https://arxiv.org/pdf/1506.04214.pdf)
> 利用雷达降雨量预测，不同雷达之间有空间联系，同一个雷达的序列数据存在时间联系，用传统CNN刻画雷达间的联系，用LSTM刻画时间联系，把LSTM中参数和输入数据的矩阵乘法替换成卷积，使得同时建模空间和时间约束，虽然输入数据依旧是标准张量，但是把时间空间结合起来。

- [Structured Sequence Modeling with Graph Convolutional Recurrent Networks (ICLR2017)](https://arxiv.org/pdf/1612.07659.pdf)
> 把时间数据和空间数据结合起来，方法有输入数据做完图卷积然后再输入LSTM，或者把LSTM中的矩阵乘法替换成图卷积。

- [Graph convolutional recurrent neural network data-driven traffic forecasting](https://arxiv.org/pdf/1707.01926.pdf)
> 上一篇论文方法的应用，道路传感器网络上下一时刻的速度预测，取得了state of the art的表现。

- [Convolutional Networks on Graphs for Learning Molecular Fingerprints (NIPS2015)](https://arxiv.org/pdf/1509.09292.pdf)
> 分子是原子及连边的图，任务是预测分子的属性，把原始方法中的hash函数替换成用一层neural network雪一个smooth的function，利用hash结果取余找index的过程替换成softmax。输入是图的形式，但是处理方法和上面论文不太一样。

- [Geometric deep learning: going beyond Euclidean data (IEEE Signal Processing Magazine 2017)](https://arxiv.org/pdf/1611.08097.pdf)
> 一篇review

下面几篇在处理dynamic graph，把传统CNN与图上CNN统一起来

- [Transfer learning for deep learning on graph-structured data (AAAI2017)](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14803/14387)

- [Graph Convolution: A High-Order and Adaptive Approach ](https://arxiv.org/pdf/1706.09916.pdf)

- [Learning Graph Convolution Filters from Data Manifold](https://arxiv.org/pdf/1710.11577.pdf)

- [Diffusion-Convolutional Neural Networks](https://arxiv.org/pdf/1511.02136.pdf)

- [Dynamic Graph Convolutional Networks](https://arxiv.org/pdf/1704.06199.pdf)

下面几篇还没有读

- [Geometric deep learning on graphs and manifolds using mixture model CNNs (CVPR2017)](https://arxiv.org/pdf/1611.08402.pdf)

- [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs (CVPR2017)](https://arxiv.org/pdf/1704.02901.pdf)

- [Protein interface prediction using graph convolutional networks (NIPS2017)](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf)

- [Gated Graph Sequence Neural Networks (ICLR2016)](https://arxiv.org/pdf/1511.05493.pdf)

