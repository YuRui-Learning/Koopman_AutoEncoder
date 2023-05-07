import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Visualize decoder setting
# Parameters
learning_rate = 0.015
learning_rate2 = 0.1
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784  # 28x28 pix，即 784 Features
n_output = 10

# hidden layer settings
n_hidden_1 = 128  # 经过第一个隐藏层压缩至256个
n_hidden_2 = 49  # 经过第二个压缩至128个
n_hidden_3 = 36
n_hidden_4 = 16

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
OUT = tf.placeholder("float", [None, n_output])
XX = tf.placeholder("float", [None, n_hidden_2])

# output
# 两个隐藏层的 weights 和 biases 的定义
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

classify_weights = {
    'layer1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'layer2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'layer3': tf.Variable(tf.random_normal([n_hidden_4, n_output])),
}
classify_biases = {
    'layer1': tf.Variable(tf.random_normal([n_hidden_3])),
    'layer2': tf.Variable(tf.random_normal([n_hidden_4])),
    'layer3': tf.Variable(tf.random_normal([n_output])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer 使用的 Activation function 是 sigmoid #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def classify(x):
    out_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, classify_weights['layer1']), classify_biases['layer1']))
    out_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(out_layer1, classify_weights['layer2']), classify_biases['layer2']))
    out_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(out_layer2, classify_weights['layer3']), classify_biases['layer3']))
    return out_layer3


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
classify_op = classify(XX)

# Prediction
y_pred1 = decoder_op
y_pred2 = classify_op
# Targets (Labels) are the input data.
y_true1 = X
y_true2 = OUT

# Define loss and optimizer, minimize the squared error
# 比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，
# 根据 cost 来提升我的 Autoencoder 的准确率
loss1 = tf.reduce_mean(tf.pow(y_true1 - y_pred1, 2))  # 进行最小二乘法的计算(y_true - y_pred)^2
# loss2 = tf.reduce_mean(tf.pow(y_true2 - y_pred2, 2))  # 进行最小二乘法的计算(y_true - y_pred)^2
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=OUT, logits=y_pred2))
# loss1 = tf.reduce_mean(tf.square(y_true1 - y_pred1))
optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1)
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
# optimizer2 = tf.train.GradientDescentOptimizer(0.05).minimize(loss2)

# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    print(total_batch)
    training_epochs = 20
    training_epochs2 = 30
    # Training cycle autoencoder
    for epoch in range(training_epochs):  # 到好的的效果，我们应进行10 ~ 20个 Epoch 的训练
        # Loop over all batches
        for i in range(total_batch):
            batch_xs1, batch_ys1 = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer1, loss1], feed_dict={X: batch_xs1})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("First Optimization Finished!")

    for epoch2 in range(training_epochs2):  # 到好的的效果，我们应进行10 ~ 20个 Epoch 的训练
        # Loop over all batches
        for i2 in range(total_batch):
            batch_xs2, batch_ys2 = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # print(batch_ys2)
            # Run optimization op (backprop) and cost op (to get loss value)
            encoder_result = sess.run(encoder_op, feed_dict={X: batch_xs2})
            _, c = sess.run([optimizer2, loss2], feed_dict={XX: encoder_result, OUT: batch_ys2})
            # Display logs per epoch step
        if epoch2 % display_step == 0:
            test_output = sess.run(fetches=classify_op, feed_dict={XX: encoder_result})
            test_acc = sess.run(tf.equal(tf.argmax(test_output, 1), tf.argmax(batch_ys2, 1)))
            test_accaracy = sess.run(tf.reduce_mean(tf.cast(test_acc, dtype=tf.float32)))  # 求出精度的准确率进行打印
            print("Epoch:", '%04d' % (epoch2 + 1), "cost=", "{:.9f}".format(c))
            print(test_accaracy)  # 打印当前测试集的精度
    print("Second Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(y_pred1, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.show()
    # 计算准确率
    # encoder_result2 = sess.run(encoder_op, feed_dict={X: mnist.test.images[:examples_to_show]})
    test_ax, test_ay = mnist.test.next_batch(1000)  # 则使用测试集对当前网络进行测试
    encoder_result2 = sess.run(encoder_op, feed_dict={X: test_ax})
    print(test_ay)
    test_output = sess.run(fetches=classify_op, feed_dict={XX: encoder_result2})
    print(test_output)
    test_acc = sess.run(tf.equal(tf.argmax(test_output, 1), tf.argmax(test_ay, 1)))
    print(test_acc)
    test_accaracy = sess.run(tf.reduce_mean(tf.cast(test_acc, dtype=tf.float32)))  # 求出精度的准确率进行打印
    print(test_accaracy)  # 打印当前测试集的精度

