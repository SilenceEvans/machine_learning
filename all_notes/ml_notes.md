

# 机器学习笔记

我们将机器学习主要分为监督学习和非监督学习。

监督学习又主要分为：

- 回归模型
- 分类模型

## 线性回归

线性模型试图学得一个通过属性（特征）的线性组合来进行预测的函数，即

![图片1](/Users/chenxiwang/Desktop/推荐系统学习笔记/screenshot/图片1.png)

一般用向量形式写成：![图片2](/Users/chenxiwang/Desktop/推荐系统学习笔记/screenshot/图片2.png)

其中![图片3](/Users/chenxiwang/Desktop/推荐系统学习笔记/screenshot/图片3.png)，是个列向量（本来应该是竖着写，此处是为了方便)。$\omega$、b习得之后，模型即得以确定。 

- 许多功能强大的非线性模型就是在线性的基础之上，通过引入层级结构或高纬映射来得到。

### 最小二乘法

是处理损失函数的算法，主要用来处理一元代价函数。

基于**<u>均方误差最小化</u>**来进行模型求解的方法称为最小二乘法（least square method）

- 主要思想是选择未知参数，使得理论值和观测值之差的平方和达到最小

​                            <img src="/Users/chenxiwang/Desktop/推荐系统学习笔记/screenshot/图片4.png" alt="图片4" style="zoom:50%;" /> 

上图中蓝色的点就是观测值，黄色的点就是理论值

#### 最小二乘法的公式推导

$E_{(\omega, b)}=\sum_{i=1}^{m}\left(y_{i}-y\right)^{2}$

> $y_{i}$代表实际值，$y$代表拟合出来的期望值

其中$y=\omega x_{i}+b$，代入上式得：

$E_{(\omega, b)}=\sum_{i=1}^{m}\left(y_{i}-\omega x_{i}-b\right)^{2}$

为求$E_{(\omega, b)}$的最小值，分别对$\omega及b$求偏导

先对${\omega}$求偏导：

$E_{\omega}^{\prime}=2\left[\omega \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right]$

再对b求偏导：

$E_{b}^{\prime}=2\left[m b+\sum_{i=1}^{m}\left(\omega x_{i}-y_{i}\right)\right]\\$

令$E_{b}^{\prime}=2\left[m b+\sum_{i=1}^{m}\left(\omega x_{i}-y_{i}\right)\right]=0$

求得:

> $b=\frac{1}{m}\left(\sum_{i=1}^{m}\left(y_{i}-\omega x_{i}\right)\right)=\bar{y}-\omega \bar{x}$

将b代入到$E_{\omega}^{\prime}=0$，得：

$\omega \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left[y_{i}-\frac{1}{m}\left(\sum_{i=1}^{m}\left(y_{i}-\omega x_{i}\right)\right)\right] x_{i}=0 \leftrightarrow \omega \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m} y_{i} x_{i}+\sum_{i=1}^{m} x_{i}(\bar{y}-\omega \bar{x})=0 \\
\omega\left(\sum_{i=1}^{m} x_{i}^{2}-m(\bar{x})^{2}\right)=\sum_{i=1}^{m} y_{i} \cdot x_{i}-m \bar{x} \bar{y}=\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)$

即可求得

> $\omega=\frac{\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)}{\left(\sum_{i=1}^{m} x_{i}^{2}-m(\bar{x})^{2}\right)}=\frac{\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{m} x_{i}^{2}-\frac{1}{m}\left(\sum_{i=1}^{m} x_{i}\right)^{2}}$

综上

> $\left\{\begin{array}{l}
> b=\frac{1}{m}\left(\sum_{i=1}^{m}\left(y_{i}-\omega x_{i}\right)\right) \\
> \omega=\frac{\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{m} x_{i}^{2}-\frac{1}{m}\left(\sum_{i=1}^{m} x_{i}\right)^{2}}
> \end{array}\right.$



### 梯度下降算法

主要用来处理多元函数

#### 首先要解决的数学问题

- 全微分的定义
- 方向导数的定义及其几何意义
- 梯度的定义及其几何意义

##### 全微分的定义

为了方便，手写公式，插入图片

![image-20211103180631219](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211103180631219.png)

##### 方向导数及梯度

![image-20211103164930509](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211103164930509.png)

![image-20211103165003322](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211103165003322.png)

> 考虑一个问题，梯度方向为什么是等值线的法线方向？
>
> <img src="/Users/chenxiwang/machine_learning/all_notes/images_of_all_notes/截屏2021-11-13 下午10.08.18.png" alt="截屏2021-11-13 下午10.08.18" style="zoom:33%;" />
>
> 我们知道，梯度方向是当$（f_{x},f_{y}）$与$（cos\alpha,cos\beta）$这两个向量同向时的方向，而方向导数为0时的方向是等值线对应到空间中的线的切线方向，方向导数为0，说明：$（f_{x},f_{y}）\cdot(cos\alpha,cos\beta)=|(f_{x},f_{y})|\cdot|(cos\alpha,cos\beta)|\cdot cos\theta=0$，说明此时$\theta = \pi/2$，此时$（cos\alpha,cos\beta）$的方向就是等值线切线的方向，而我们要明确的一点是，**梯度的方向一直是与$(f_{x},f_{y})$同向的**。所以，梯度的方向就是与等值线切线方向垂直的法线方向。
>
> 此处的逻辑证明应该是：
>
> 方向导数为0的方向是等值线的切线方向，而方向导数为0的方向又是和梯度方向垂直的方向，故梯度的方向就是和等值线切线方向垂直的方向。



#### 梯度下降求解线性回归

![image-20211103165502084](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211103165502084.png)

对于上图红框中的内容主要的疑问为：

- 红框中":= "是什么意思？
- 为什么$\theta$ 的更新用的是减号而不是加号？

##### := 的意思

:= 应该是来自于吴恩达机器学习视频中的符号表示，根据吴老师的介绍，:= 意思就是右边的$\theta$减去$\alpha$乘以$\theta$的偏导数等于左边的$\theta$，这是一个对$\theta$进行更新的过程。

这儿对于我们熟悉变成的人来说可能有点疑惑，平时我们更新一个变量本身不就是 a = a + b或者

a += b这么去更新的吗？为什么这儿更新会变成 := ？

这个地方我个人更愿意理解为 := 是一种严格的数学语言，比如我们在数学中就不能说 4 = 4 + 1，这样是错误的，只不过这个地方把4换成了变量而已。其实我们在具体的Python或者其他语言的编码当中，此处更新$\theta$当然还是用的$\theta$ = $\theta$ - 它的偏导。

##### 为什么更新用的是减号

这个可能需要联系梯度的定义，数学中这么定义梯度：

空间中某点的梯度就是该点方向导数最大的方向。

注意：梯度是向量，向量是有方向的，这个方向是导数最大的方向，即函数值增大最快的方向，而我们的梯度下降，是在求这个误差函数的函数值能最快变得最小的那个方向，换而言之，这个方向是与梯度向量方向在一条直线上但完全相反的方向，就像：

​															$\uparrow$ 梯度方向，函数值变大最快的方向
​															$\downarrow$ 梯度下降方向，函数值变小最快的方向

我们知道假设对于一个 $Z=(x,y)$的二元函数来说，它在$(x,y)$点的梯度就是向量$(\frac{\partial Z}{\partial x},\frac{\partial Z}{\partial y})$,此时要沿着梯度的反方向变化，这就是“—”而不是“+”的原因，当然是$Z=(x,y)$点的横纵坐标同比例的（这也是乘$\alpha$再减的原因）加上梯度相反方向向量，想像一个空间点沿着梯度相反方向移动，具体移动多少，看步长$\alpha·\frac{\partial Z}{\partial x}$和$\alpha·\frac{\partial Z}{\partial y}$的大小。

画个图：

![image-20211103174931435](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211103174931435.png)

上图反应的是沿梯度方向变化，及增大，对比可以了解沿梯度反方向下降的过程。

![image-20211103175402466](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211103175402466.png)

##### 上述两个框中的表达式为什么不一样？

这个其实比较好回答，首先要弄清楚我们这个函数求偏导的过程中谁是未知数，谁是已知数，我们是在对谁求偏导？

在上述的表达式中，$x，y$其实是我们样本中已知的数据，未知数是表达式里的众多$\theta$，对于有$x$这个系数的未知数$\theta$，对$\theta$求完偏导之后当然还应该乘以这个系数$x$,而对于没有系数的未知数$\theta$，求完导后$\theta$成了1，当然就没有$x$了。

## 分类

### KNN近邻算法

### 逻辑斯蒂回归（logistic regression）

思考以下两个问题：

> 1.为什么要使用logistic regression？
>
> 2.是一个分类算法，为什么要用到回归这样的字眼？

![截屏2021-11-12 下午4.21.37](/Users/chenxiwang/Desktop/推荐系统学习笔记/screenshot/截屏2021-11-12 下午4.21.37.png)

#### 为什么要使用logistic regression？

对于上图中，我们依据肿瘤的大小来对肿瘤是否是良性进行分类，暂时使用线性函数对样本进行拟合，洋红色是拟合出来的曲线，此时得到了一个比较好的拟合效果。我们以拟合函数值0.5作为分类的条件，输入样本特征，函数值小于0.5时属于良性肿瘤，函数值大于0.5时属于恶性肿瘤。

就第一次拟合出来的洋红色曲线，似乎我们已经得到了比较好的分类效果。但考虑下面这样一个问题：

**如果此时样本中有一个或者几个肿瘤尺寸特别的大（最右侧的红色样本点），我们的拟合函数可能为了拟合到这样的肿瘤样本点，函数图像就会从洋红色曲线变成蓝色的曲线，但是这是我们依旧使用之前的函数值大于0.5时是恶性肿瘤，函数值小于0.5时是良性肿瘤，会出现什么问题呢？**

我们对着函数图像就可以知道，原先肿瘤大小在拟合函数上取值大于0.5的点，此时在蓝色曲线上变成了小于0.5的点，很可能就因为我们这样的一次拟合导致我们很多之前归为恶性肿瘤的样本变成了良性肿瘤。而且我们不可能总是去因为一些比较极端特殊的指标去改变我们进行分类的指标，比如就因为这样一个极端的点，我们将我们的指标从大于0.5是恶性肿瘤变成指标大于0.3是良性肿瘤，这显然是不合理的。

#### 是一个分类算法，为什么要用到回归这样的字眼？

吴恩达老师在视频中说logistic regression用到回归这样的字眼是因为一些历史问题，但本质上是分类问题。

但通过上述这样一个例子，我们可以发现它和线性回归可能之间存在着某种联系，是因为线性回归做不到比较好的分类，从而对其进行的一个改进，改进成为一个适用于二分类或者说是多分类的一个算法。

#### 模型函数与决策边界

模型函数：

​								$g(z)=\frac{1}{1+e^{-z} }$，其中$z=h_{\theta}(x)$

​							$h_{\omega}(x)=\theta ^{T} X$，其中$\omega ^{T}=\left [  \theta _{0} ,\theta _{1} ,\theta _{2} ...\theta _{i} \right ] $，$X=\begin{bmatrix}\\1 \\x_{1}\\x_{2}\\...\\x_{i}\end{bmatrix}$

$g(z)$就是我们常说的sigmoid函数，或者叫logistic函g

函数图像：

<img src="/Users/chenxiwang/machine_learning/all_notes/images_of_all_notes/截屏2021-11-12 下午8.12.06.png" alt="截屏2021-11-12 下午8.12.06" style="zoom: 50%;" />



$z=h_{\omega}(x)$是我们所说的决策边界。

> 决策边界不是训练集固有的属性，使我们假设出来的，拟合出来的这样一个函数

下图中红色圆就是我们拟合出的这个决策边界，依据这个决策边界我们将样本分为了两类。

![截屏2021-11-12 下午7.44.35](/Users/chenxiwang/Library/Application Support/typora-user-images/截屏2021-11-12 下午7.44.35.png)

#### 代价函数

之前线性回归的代价函数:

​					           				$J(\theta )=\frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^{2}$

上式中的$h_{\theta}(x)$在逻辑回归的模型中对应的函数是$g(z)$，即$g(h_{\theta}(x))$，对于这样的一个sigmoid函数如果我们也按照之前的代价函数代入的话，即$J(\theta )=\frac{1}{m} \sum_{i=1}^{m} (g(z^{(i)})-y^{(i)})^{2}$，这个函数是其实是一个非凸函数，可能类似于下图：

![截屏2021-11-12 下午8.31.48](/Users/chenxiwang/machine_learning/all_notes/images_of_all_notes/截屏2021-11-12 下午8.31.48.png)

对于这样一个函数，我们是无法求出其全局的最小值的，它只有局部的最优解。

> 数学中的凸函数和凹函数和我们此处的称法**有所区别**，类似开口向上的二次函数，我们数学中称它为**凹函数**，或者**向上凹**的函数，而我们此处将这种有最值的函数都称为**凸函数**。
>
> - 为什么线性回归的代价函数就能知道它是凸函数？
>
>   联系数学中关于函数凹凸性的判别方法:
>
>   ${f}''(x)>0$，函数是凹函数，有最小值
>
>   ${f}''(x)<0$，函数是凸函数，有最大值
>
>   ？

故给出下面逻辑回归的代价函数：

​						$Cost(g(z),x) = \left\{\begin{array}{l}
​									-\log_{}{g(z)} \qquad \qquad \ \ y = 1\\
​									-\log_{}{(1-g(z)) \qquad y = 0} \end{array}\right.$

对应的不同的损失函数图像为：

$-\log_{}{g(z)},y = 1$部分为：

<img src="/Users/chenxiwang/machine_learning/all_notes/images_of_all_notes/截屏2021-11-12 下午9.11.00.png" alt="截屏2021-11-12 下午9.11.00" style="zoom:67%;" />

$-log(1-g(z)),y=0$部分为：



<img src="/Users/chenxiwang/Library/Application Support/typora-user-images/截屏2021-11-12 下午9.04.51.png" alt="截屏2021-11-12 下午9.03.54" style="zoom: 67%;" />

将上述代价函数归一化处理：

​																$Cost(g(z),x) =-y\cdot\log_{}{g(z)}-(1-y)\cdot\log_{}{(1-g(z))}$

对于上述函数我们就可以利用梯度下降算法实现梯度更新。

##### 一般梯度下降算法

代价函数对$\omega$求偏导的过程如下：

> 首先要说明的是，虽然此处们用的对数表示是log，但其实换成自然对数ln效果是一样的。

先对代价函数进行化简：

$
{Cost\text{(}g\text{(}z\text{)},x\text{)}=-y \cdot ln\mathop{{}}\nolimits^{{g \left( z \right) }}-ln\mathop{{}}\nolimits^{{ \left( 1-g \left( z \left)  \right) \right. \right. }}+y∙ln\mathop{{}}\nolimits^{{ \left( 1-g \left( z \left)  \right) \right. \right. }}}\\
{Cost\text{(}g\text{(}z\text{)},x\text{)}=y∙ln\mathop{{}}\nolimits^{{\frac{{1-g \left( z \right) }}{{g \left( z \right) }}}}-ln\mathop{{}}\nolimits^{{ \left( 1-g \left( z \left)  \right) \right. \right. }}}\\
{g \left( z \left) =\frac{{1}}{{1+\mathop{{e}}\nolimits^{{-wx}}}}\right. \right. }\\
{1-g \left( z \left) =\frac{{\mathop{{e}}\nolimits^{{-wx}}}}{{1+\mathop{{e}}\nolimits^{{-wx}}}}\right. \right. }\\
{Cost\text{(}g\text{(}z\text{)},x\text{)}=y∙ln\mathop{{}}\nolimits^{{\frac{{1-g \left( z \right) }}{{g \left( z \right) }}}}-ln\mathop{{}}\nolimits^{{ \left( 1-g \left( z \left)  \right) \right. \right. }}}\\
{Cost\text{(}g\text{(}z\text{)},x\text{)}=y∙ \left( -wx \left) -ln\mathop{{}}\nolimits^{{\frac{{\mathop{{e}}\nolimits^{{-wx}}}}{{1+\mathop{{e}}\nolimits^{{-wx}}}}}}\right. \right. }\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }=y∙ \left( -wx \left) - \left( -wx-ln\mathop{{}}\nolimits^{{1+\mathop{{e}}\nolimits^{{-wx}}}} \right) \right. \right. }\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }=-ywx+wx+ln\mathop{{}}\nolimits^{{1+\mathop{{e}}\nolimits^{{-wx}}}}}\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }=w \left( x-xy \left) +ln\mathop{{}}\nolimits^{{1+\mathop{{e}}\nolimits^{{-wx}}}}\right. \right. }{}\\$

再对代价函数求偏导（对$w$求偏导）：

${Cos{t \prime }\text{(}g\text{(}z\text{)},x\text{)}=\frac{{ \partial  \left( Cost \right) }}{{ \partial w}}}\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }= \left( 1-y \left) x-\frac{{xe\mathop{{}}\nolimits^{{-wx}}}}{{1+e\mathop{{}}\nolimits^{{-wx}}}}\right. \right. }\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }={ \left[ { \left( 1-\frac{{e\mathop{{}}\nolimits^{{-wx}}}}{{1+e\mathop{{}}\nolimits^{{-wx}}}} \left) -y\right. \right. } \right] }∙x}\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }= \left( \frac{{1}}{{1+e\mathop{{}}\nolimits^{{-wx}}}}-y \left) ∙x\right. \right. }\\
{\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }\text{ }= \left( g \left( z \left) -y \left) x\right. \right. \right. \right. }\\
{}$

其实对比之前线性回归代价函数求偏导的结果：

$\frac{\partial J(\theta)}{\partial \theta}==\frac{2}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})x^{(i)} $

我们可以看出，他们的形式是差不多一样的：

$\frac{\partial J(\omega)}{\partial \omega}==\frac{2}{m} \sum_{i=1}^{m} (g(z(x^{(i)}))-y^{(i)})x^{(i)}$

求出偏导数，我们便可以进行梯度下降：

$\omega := \omega - \frac{\partial J(\omega)}{\partial \omega}$

##### 更优化的梯度下降算法——BFGS和L-BFGS

#### 多元分类

## 过拟合问题

线性回归曲线：

![截屏2021-11-13 下午4.34.00](/Users/chenxiwang/machine_learning/all_notes/images_of_all_notes/截屏2021-11-13 下午4.34.00.png)

逻辑回归曲线：

![截屏2021-11-13 下午4.34.32](/Users/chenxiwang/machine_learning/all_notes/images_of_all_notes/截屏2021-11-13 下午4.34.32.png)

对于线性回归或者逻辑回归问题，我们都需要拟合一条曲线出来，拟合的这条曲线，通常会存在下面这几个情况：

- 欠拟合（Underfit）或者高偏差（High bias）

  样本点没能很好的跟我们的曲线进行拟合，存在较大的偏差

- 刚好合适（Just right）

- 过拟合（Overfit）或者高方差（High variance）

  曲线很好的拟合了我们的训练样本，但是因为拟合的太好，模型在实验集中不会有太好的结果，**泛化**能力较差

  > 方差是在概率论和统计方差衡量[随机变量](https://baike.baidu.com/item/随机变量/828980)或一组数据时离散程度的度量。概率论中方差用来度量[随机变量](https://baike.baidu.com/item/随机变量/828980)和其[数学期望](https://baike.baidu.com/item/数学期望/5362790)（即[均值](https://baike.baidu.com/item/均值/5922988)）之间的偏离程度。统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的[平均数](https://baike.baidu.com/item/平均数/11031224)。在许多实际问题中，研究方差即偏离程度有着重要意义。
  >
  > 方差是衡量源数据和期望值相差的度量值。

### 过拟合问题的处理

通过绘图来决定多项式的方法似乎有用，但实际操作起来是行不通的，我们无法通过图形的变化去找到一个特别合适的多项式。另外实际上，当多项式的项数变得多起来之后，绘制图形也会变得逐渐困难起来。

所以往往通过下面几种方式解决过拟合问题。

**1、减少样本特征项**

- 手动的选择哪些样本特征需要保持

- 模型选择算法（Model selection algorithm）

  模型选择算法虽然可以自动的帮我们处理哪些特征保留，哪些特征项废弃，但有时候自动筛选掉过着保留的特征项不一定是我们想要的

**2、正则化（Regularization）**

- 保留所有的特征值，但是缩放参数$\theta_{j}$的值
- 在我们有很多特征值的时候表现不错，每一个特征项为我们结果的预测都贡献出了一点效果



