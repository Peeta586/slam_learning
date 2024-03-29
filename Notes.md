# SLAM 笔记
[TOC]

## 1. slam基本组成
- 前端： 视觉里程计
- 后端： 优化
- 回环： loop closing
- 建图： mapping

### 1.1 视觉里程计
**1. 什么是里程计**
在里程计问题中，我们希望测量一个运动物体的轨迹。这可以通过许多不同的手段来实现。例如，我们在汽车轮胎上安装计数码盘，就可以得到轮胎转动的距离，从而得到汽车的估计。或者，也可以测量汽车的速度、加速度，通过时间积分来计算它的位移。完成这种运动估计的装置（包括硬件和算法）叫做里程计（Odometry）。

**2. 里程计的特性**
里程计一个很重要的特性，是它只关心局部时间上的运动，多数时候是指两个时刻间的运动。当我们以某种间隔对时间进行采样时，就可估计运动物体在各时间间隔之内的运动。由于这个估计受噪声影响，==*先前时刻的估计误差，会累加到后面时间的运动之上，这种现象称为漂移（Drift）*==。

**3. 什么是视觉里程计？**
　　视觉里程计VO的目标是根据拍摄的图像估计相机的运动。它的主要方式分为特征点法和直接方法。其中，特征点方法目前占据主流，能够在噪声较大、相机运动较快时工作，但地图则是稀疏特征点；直接方法不需要提特征，能够建立稠密地图，但存在着计算量大、鲁棒性不好的缺陷。

>我们知道在汽车中有一个里程计，记录着汽车行驶的距离，可能的计算方式就是通过计算车轮滚动的次数乘以轮子的周长，但是里程计会遇到精度的问题，例如轮子打滑，随着时间的增加，误差会变得越来越大。另外在我们机器人和视觉领域，不仅仅要知道行驶的距离，而且要知道机器人行驶的整个轨迹（机器人每个时刻的位置和姿态）我们记着在时间t时刻机器人的位置和姿态信息是(xt,yt,zt,ψt,χt,ϕt)，其中xt,yt,zt表示机器人在世界坐标系中的位置信息，ψt,χt,ϕt表示机器人的姿态，分布表示为roll(ψt), pitch(χt),yaw(ϕt)

>确定机器人轨迹的方法有很多，我们这里主要讲述的是视觉里程计，正如维基百科所述，我们将一个摄像头（或多个摄像头）刚性连接到一个移动的物体上（如机器人），通过摄像头采集的视频流来确定相机的6自由度，如果使用1个摄像头，则称为单目视觉里程计，如果使用两个（或者更多）摄像机，则称为立体视觉里程计

**4. 主要实现方法**
- 特征点方法目前占据主流
- 直接法

### 1.2. 后端-优化
- 从带有噪声的数据中估计最优轨迹与地图
- 最大后验概率
- 滤波器
- 图优化

当有噪声时，经过优化算法，如：滤波器，图优化等消除噪声

### 1.3. 回环检测
- 检测相机是否达到过之前的位置
- 判断与之间位置的差异
- 计算图像间相似性
- 词袋模型

### 1.4. 建图
- 导航，规划，通信，交互，可视化
- 度量地图，拓扑地图
- 稀疏地图，稠密地图

## 2. slam的数学描述
### 2.1 运动方程
$$x_k = f(x_{k-1}, u_k, w_k)$$
表示状态统计; $x_{k-1}$ 表示k-1时刻设备的位置状态，$x_k$是k时刻的位置状态， 我们可以用上面的公式表示，
这个动作过程我们是知道的，因为是我们输入一个命令给机器人什么的进行运动，如向前多少米这样的已知信息$u_k$，这样我们用f表示上一时刻，到下一时刻的变换，$w_k$是噪声。
(例如： f表示我们执行右转的一个操作里程计记录，$x_{k-1}$经过右转里程计信息及噪声得到下一时刻位置）

### 2.2 观测方程
$$z_{k,j} = h(y_j, x_k,v_{k,j})$$
这个表示在k时刻对j对象进行采集，这样我们获取到该时刻的这个对象j，结合上一时刻的对象位置j，可是实现建图;
在某个位置对某个对象观测表示为测量， 这个也会有噪声，$v_{k,j}$表示噪声

### 2.3 总结
已知量$u_k$, $z_{k,j}$是知道的，我们slam的目的就是求，$x_k$（位置状态）用于定位， $y_j$（某对象的位置）用于建图


## 3. 刚体运动
- 点与坐标系
- 旋转矩阵
- 旋转向量和欧拉角
- 四元数
- 实战： Eigen

### 3.1 自己的位置如何表达-点与坐标系
-2D: (x,y,$\theta$)， $\theta$ 是朝向-旋转角

**矩阵基础运算**
$a \times b = -b \times a = a^\# * b, 其中$$a^{\#} $$ 表示反对称矩阵; 和对称矩阵类似，只是对称的地方元素互为正负
$a^T * b = b^T * a$

### 3.2 坐标系之间的转换
坐标系转换-旋转： 利用旋转矩阵进行操作
