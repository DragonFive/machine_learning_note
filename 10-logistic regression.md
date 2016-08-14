以下是台大林轩田老师讲的机器学习基石第10课的学习笔记 。
# 软性二值分类(soft binary classification)
##目标函数 
![gif.gif](http://upload-images.jianshu.io/upload_images/454341-9bb5f38f2192e540.gif?imageMogr2/auto-orient/strip)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-6a1db3f52380e71b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里我们的二值分类和硬性二值分类的数据是一样的，但是目标函数是不一样的。而软性二值分类所真正需要的数据是跟目标函数一样的概率，但是我们收集的数据却是分类的结果。

## logistic hypothesis
对于提取的特征向量： 
![gif.gif](http://upload-images.jianshu.io/upload_images/454341-1e860f4fd4898aa4.gif?imageMogr2/auto-orient/strip)
计算各个分量的加权分数，但我们需要把这个分数转化为0-1之间的概率值。（因为我们的目标函数是这个）
用到的转换函数叫logistic函数

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-b93f87a9b6c4991c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样我们的logistic hypothesis就是:

![gif.gif](http://upload-images.jianshu.io/upload_images/454341-559959ad3e428044.gif?imageMogr2/auto-orient/strip)

而其中的的logistic function(sigmoid函数就一种)可以为：
![](http://latex.codecogs.com/gif.latex?%5Ctheta%28s%29%3D%20%5Cfrac%7B%20e%5Es%7D%7B1+e%5Es%7D%3D%5Cfrac1%7B1+e%5E%7B-s%7D%7D)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-0ec773543ad62f05.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
sigmoid型函数表示是一个s型的函数。

# logistic 回归
## 作法
用 ![](http://latex.codecogs.com/gif.latex?h%28x%29%3D%5Cfrac1%7B1+exp%28-w%5ETx%29%7D) 来近似目标函数 ``` f(x)=P(y|x) ```

## error measure错误衡量

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-6bf978134514d324.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们这里也要找一个Ein来minimise一下 ，当我们的目标是一个概率p的时候，我们可以用一个特殊的方式。
这个方式就是**最大似然估计**的方法，我们假设目标函数为：
![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-8b64b143908b6f42.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

则对于一个数据，它取各个值的概率分别为：

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-fcfc37622bc9cd25.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么我们可以从数据中取出N个样本(in sample),观测它们的分布，我们想要达到的目标是我们的目标函数能够让取出这N个观测的概率尽可能的大，这个就是最大似然估计得到最优化的方法。


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-dbaa1072e2486537.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用f(x)替换成


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-40ad81df941d30db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用我们的hypothesis替换f:

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-04ab9163a575e27d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-8324a1b92800a34e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
让这个可能性最大的g就是我们要找的g

现在我们发现这个s型的logistic函数有对称性 

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-96ae658348d873de.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

所以我们可以用这个性质来简化优化函数，因为p(xi)对于所有的h是一样的，所以没什么关系 


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-d19c1c51847e885c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
 
然后我们用我们的hypothesis的定义式子来替换这个h，要找likelihood的最大值，我们把连乘通过取对数换成连加，通过带入logistic函数最终得到Ein最小化的形式。这个error 衡量我们叫交叉熵错误（信息熵的概念）。
![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-7ee8712992ff68f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 最优化
对这个Ein 求梯度为0 的w的值


![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-97faa9bf736908aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-002bd16bbc0dd775.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

要想让这个Ein的梯度小到接近0，就不断的尝试 启发式搜索 、**迭代优化（iterative optimization）**


v 是方向 η是步频

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-faee68d22334a08c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
 
每一步都要用贪心的策略，找一个下降最快的方向 
![每一步](http://upload-images.jianshu.io/upload_images/454341-2cb39125c35e727c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个优化对象不是线性的，我们应该使用泰勒展开的形式，把公式近似替代为线性的形式

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-84f540bc714e3bae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 梯度下降法 gradient descent 

v的方向取梯度的反方向 

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-4868470971309faf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

η 应该和梯度的大小成比例，这样才能最终收敛。这样和v的分母抵消，最后形成定值学习率(fixed learning rate )

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-5ecd4943533a2d37.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面是logistic 回归算法用梯度下降法做优化 

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-4d23675cfa38ee4c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/454341-e22dcce2cb73fd11.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 其它资料 

[logistic回归](http://blog.csdn.net/mosbest/article/details/52163246)



