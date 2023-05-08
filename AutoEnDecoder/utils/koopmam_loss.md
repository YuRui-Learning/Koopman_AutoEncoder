# KOOPMAN LOSS

## 编码器损失

### 网络结构

方法Deep EDMD:在Deep EDMD中，编码器的结构为5层，结构选择为[n 32 64 L−n L−n]。解码器的结构设置为[L 128 64 32 n]。模拟时将L设为13。模拟中使用的所有参数列于表2。对于ω = 1/√a，所有权值以均匀分布初始化，将每个权值限制在[−ω ω]范围内，其中a为网络层数[56]。

### 多步预测损失函数

![image-20230505204017227](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204017227.png)

其中rx,p是第p步预测误差，K[p]是xt开始超前p步状态

![image-20230505204034231](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204034231.png)

故沿p个时间步长预测误差和，loss定义为

![image-20230505204048913](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204048913.png)

### 观测空间的预测误差

最小化提升空间中的状态演化和从真实动态中提升状态序列映射的误差

![image-20230505204227520](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204227520.png)

###  解码器的损失函数

![image-20230505204200939](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204200939.png)

同时为了保证鲁棒性，采用无穷范数作为损失函数：

![image-20230505204136348](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204136348.png)

### 整体误差

![image-20230505204242263](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204242263.png)

![image-20230505204256083](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204256083.png)

a1-a6为权重，||θ||^2其为为l2正则化项，用于避免过拟合

![image-20230505204315758](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204315758.png)

4、模型输入X和U，但是这个是p的X和P-1的U，那这个是从1-P的数据，还是第p个数据，如果是1-p的话每次数据都不一样吗？

5、用P步预测后面数据和真实的后面数据进行比较，输入到loss中去。

6、神经网络中的w和b是不是可以理解为A和B，一个是权重，一个是偏置，正则化θ更新。



## MLP损失

### 网络结构

MLP方法的结构选择为[n + m 32 64 128 128 64 32 n]。在比较中考虑了两种情况。在第一种场景中，所有参数都作为需要学习的优化变量，而在第二种场景中，在Deep EDMD和mlp中分别增加了一个隐藏层，权重和偏差在整个训练过程中以正态分布随机更新。用θM参数化MLP的权值和偏置，给出MLP的损失函数为

![image-20230505204403664](C:\Users\18423\AppData\Roaming\Typora\typora-user-images\image-20230505204403664.png)

第一项表示多步预测损失。第二项是无限范数，用于惩罚预测的最大误差，最后一项是l2正则化项，用于避免过度拟合



## LSTM

LSTM的结构包含三个LSTM单元，每个LSTM单元的输入输出维数相同，且n值相等。第二个和第三个LSTM单元分别将第一个和第二个LSTM单元的输出作为输入。最后一个LSTM单元与一个结构为[n 128 128 n]的MLP连接，用于预测下一个状态。这三个LSTM单元采用普通的LSTM单元[57]，它由一个单元、一个遗忘门、一个输入门和一个输出门组成。LSTM中MLP的每一层都有一个ReLU作为激活函数，除了最后一层是线性的。在训练中，设计的LSTM和MLP方法具有(29)中相同的损失函数和表II中的超参数，如步长p和学习率。

### 代码里面用到的损失

自编码损失 loss1

```
if params['relative_loss']:
    loss1_denominator = tf.reduce_mean(tf.reduce_mean(tf.square(tf.squeeze(x[0, :, :])), 1)) + denominator_nonzero
else:
    loss1_denominator = tf.to_double(1.0)

mean_squared_error = tf.reduce_mean(tf.reduce_mean(tf.square(y[0] - tf.squeeze(x[0, :, :])), 1))
loss1 = params['recon_lam'] * tf.truediv(mean_squared_error, loss1_denominator)
```

动力学/预测损失函数 loss2

```
loss2 = tf.zeros([1, ], dtype=tf.float64)
if params['num_shifts'] > 0:
    for j in np.arange(params['num_shifts']):
        # xk+1, xk+2, xk+3
        shift = params['shifts'][j]
        if params['relative_loss']:
            loss2_denominator = tf.reduce_mean(
                tf.reduce_mean(tf.square(tf.squeeze(x[shift, :, :])), 1)) + denominator_nonzero
        else:
            loss2_denominator = tf.to_double(1.0)
        loss2 = loss2 + params['recon_lam'] * tf.truediv(
            tf.reduce_mean(tf.reduce_mean(tf.square(y[j + 1] - tf.squeeze(x[shift, :, :])), 1)), loss2_denominator)
    loss2 = loss2 / params['num_shifts']
```

线性损失函数 loss3

```
loss3 = tf.zeros([1, ], dtype=tf.float64)
count_shifts_middle = 0
if params['num_shifts_middle'] > 0:
    # generalization of: next_step = tf.matmul(g_list[0], L_pow)
    omegas = net.omega_net_apply(params, g_list[0], weights, biases)
    next_step = net.varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'],
                                     params['num_complex_pairs'])
    # multiply g_list[0] by L (j+1) times
    for j in np.arange(max(params['shifts_middle'])):
        if (j + 1) in params['shifts_middle']:
            if params['relative_loss']:
                loss3_denominator = tf.reduce_mean(
                    tf.reduce_mean(tf.square(tf.squeeze(g_list[count_shifts_middle + 1])), 1)) + denominator_nonzero
            else:
                loss3_denominator = tf.to_double(1.0)
            loss3 = loss3 + params['mid_shift_lam'] * tf.truediv(
                tf.reduce_mean(tf.reduce_mean(tf.square(next_step - g_list[count_shifts_middle + 1]), 1)),
                loss3_denominator)
            count_shifts_middle += 1
        omegas = net.omega_net_apply(params, next_step, weights, biases)
        next_step = net.varying_multiply(next_step, omegas, params['delta_t'], params['num_real'],
                                         params['num_complex_pairs'])

    loss3 = loss3 / params['num_shifts_middle']
```

关于自动编码器损耗和一步预测损耗的INF范数 loss_Linf

```
if params['relative_loss']:
    Linf1_den = tf.norm(tf.norm(tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf) + denominator_nonzero
    Linf2_den = tf.norm(tf.norm(tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf) + denominator_nonzero
else:
    Linf1_den = tf.to_double(1.0)
    Linf2_den = tf.to_double(1.0)

Linf1_penalty = tf.truediv(
    tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf1_den)
Linf2_penalty = tf.truediv(
    tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf2_den)
loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)
```