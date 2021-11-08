# machine_learning
记录机器学习相关的内容，本仓库主要包括如下部分：
## CodeAndNotes
- 类似最小二乘法、梯度下降算法相关知识
- 算法学习过程中的个人心得，所需要的高等数学、线性代数方面知识的补充，算法中涉及到的理论公式的推导（每个算法中都有独立的笔记目录，里面包括算法中主要的公式推导过程）
- 算法的代码实现（Python）
## books
- 机器学习相关的一些书籍（西瓜书及配套的公式推导书籍，持续更新中...）

## 关于配置环境的相关问题

### 一、Mac os (Apple sillicon/M1)sklearn库安装不成功的解决方案

#### 1.安装conda环境并配置镜像源

对于Mac m1安装完Python之后直接使用pip install sklearn命令安装sklearn时出现错误属于正常情况，sklearn的官方介绍也说在Mac m1上安装会产生一些冲突问题，所以强烈建议配置conda环境。

sklearn的官方网站关于安装部分的介绍：https://scikit-learn.org/stable/install.html，关于Mac m1上安装sklearn的主要部分：

> Note that in order to avoid potential conflicts with other packages it is strongly recommended to use a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
>
> The recently introduced `macos/arm64` platform (sometimes also known as `macos/aarch64`) requires the open source community to upgrade the build configuration and automation to properly support it.
>
> At the time of writing (January 2021), the only way to get a working installation of scikit-learn on this hardware is to install scikit-learn and its dependencies from the conda-forge distribution, for instance using the miniforge installers:https://github.com/conda-forge/miniforge

看了https://github.com/conda-forge/miniforge的介绍（个人感觉是一个专门用在Mac m1上能安装scikit-learn的conda环境），文档中提示我们如果你Mac上安装了Homebrew（Homebrew的作用及安装自行查找）的话，直接使用命令安装：

```shell 
brew install miniforge
```

注意安装时conda安装在了哪个目录，我的安装在了：

/opt/homebrew/Caskroom/miniforge

等全部安装完成之后就可以使用conda命令安装对应的包了，但是这时安装因为是从国外的一些网站下载包，速度有时候会比较慢，可能因为下载时间过长报以下错误：

```java
CondaError: Downloaded bytes did not match Content-Length
```

建议配置国内镜像源，我选择清华的镜像源，执行以下命令：

```conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/
```

之后就可以使用conda命令进行安装了，例如：

```shell 
conda install sklearn
```

这样虽然看此时包的下载安装不成问题了，但我在使用Pycharm时依然不能导入相关的包，下面给出我的解决方案。

 #### 2.Pycharm中该如何进行配置

打开Pycharm在菜单栏，按以下顺序分别点击对应按钮：

Pycharm——preferences——project，然后选择齿轮按钮

![image-20211108220224170](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211108220224170.png)

选择Add

![截屏2021-11-08 下午10.03.01](/Users/chenxiwang/Desktop/截屏2021-11-08 下午10.03.01.png)

选择Conda Environment，然后选择Existing Environment配置对应目录

![image-20211108221057747](/Users/chenxiwang/Library/Application Support/typora-user-images/image-20211108221057747.png)

配置Interpreter目录：

按照我的理解，这时候因为我们已经安装了conda的环境，所以是需要此时配置的是Exist Environment，而解释器我们要选择conda环境里带的，而不是之前电脑上存在的。

这个解释器存在于：/opt/homebrew/Caskroom/miniforge下，可能因为这个适用于Mac m1的conda环境中只有Python3.9解释器，所以也只有这一个解释器可供选择。

==**注：此处对于Python版本选择有特殊要求的项目可能要重点关注一下是否存在其他的解决方案。**==

选择OK之后可以发现，项目中已经可以自动导包了。
