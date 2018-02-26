
'''
代码仅仅用做学习，别无他用
使用SoftMaxRegression对MNIST数据集中的数据进行识别
SoftmaxRegression会对每一种类别估算出一个概率，最后取概率最大的那个数字作为模型的输出结果
2018.2.25 晴
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot= True)
print('数据导入完毕！\n')
'''
MNIST　中的数据
在 one_hot = True 的时候表示对数据的labes进行one_hot编码
one_hot编码:[0,0,0,0,0,1,0,0,0,0]第五个数据是1，所以这个编码表示的label就是 5
训练数据集：
mnist.train.images.shape: 55000, 784
mnist.train.labels.shape: 55000, 10
测试数据集：
mnist.test.images.shape: 10000,784
mnist.test.labels.shape: 10000,10
验证数据集：
mnist.validation.images.shape:5000,784
mnist.validation.labels.shape:5000,10
'''
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# -- 创建一个新的 InteractiveSession
# -- 这个命令会将这个session注册为默认的session
sess = tf.InteractiveSession()
# --创建一个placeholder 第一个参数表示数据类型
# --第二个参数表示 tensorde shape None表示不限制输入的数据条数
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))  # 这里是将数据中的每一个像素点的值作为一个feature进行运算
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数 loss function
# 这里我们采用的是Cross-entryopy 交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])
# -- tf.reduce_mean() 用来对每个batch数据求平均值
# -- tf.reduce_sum() 就是求和
# -- reduction_indictices=[1]表示将得到的矩阵按照行进行相加求和，得到新的矩阵
# -- reduction_indictices=[0]表示将得到的矩阵按照列进行相加求和，得到新的矩阵
cross_entry = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entry)
tf.global_variables_initializer().run()

for i in range(1000):
    # 返回一个大小为[batch_size, 784]的图片数据
    # 在将所有的数据都取完之后，MNIST的数据会reshuffle。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
    if i % 50 == 0:
        # -- tf.equal() 是比较两个矩阵或者向量中元素是否相等 得到一个只包含True或者False的矩阵
        # -- tf.cast() 将bool类型的数据转换成 0，1的数据
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
