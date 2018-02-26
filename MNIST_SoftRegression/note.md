这是一个没有隐藏层的最浅的神经网络，整个流程可以分为四步：

1.定义算法公式，也就是神经forward时的计算

>这里我们采用的算法公式是：y=softmax(Wx + b)

    y = tf.nn.soft.softmax(tf.matmul(x, W) + b)

2.定义loss,选定优化器，并指定优化器优化loss
>这里的loss function我们采用的是： Cross-entropy(交叉熵)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))
>关于reduction_indictices:

![images](https://pic2.zhimg.com/v2-c92ac5c3a50e4bd3d60e29c2ddc4c5e9_r.png)

> 优化器选用的是SGD（随机梯度下降）

    # 0.5表示学习效率
    train_step= tf.train.GradientDescentOptimizer(0.5).mininize(cross_entropy)

3.迭代对数据进行训练

>mnist.train.next_batch(100)会返回一个大小为[batch_size, 784]的图片数据

>在将所有的数据都取完之后，MNIST的数据会reshuffle,然后再继续取。

    for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
4.在测试集或者验证集上对准确率进行评测
>评测的时候就是要计算估计的准确率
        # tf.argmax(y, 1) 表示在行找出每一行中最大数据所在的位置
        # 在MNIST数据集上 tf.argmax(y,1)+1 就表示图片对应的数字
        # tf.equal() 比较两个矩阵对应位置的元素是否相等,返回一个只有True和 Flalse的矩阵
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # tf.cast(correct_prediction,tf.float32)将bool类型的数据转换tf.float32类型的数据
        # 然后再对这些数据(一个batch_size大小)进行求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 以下两个语句是相等的
        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
