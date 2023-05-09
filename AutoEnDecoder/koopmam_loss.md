# KOOPMAN

## autoEnDecoder_Pytorch.py
主文件，现在主要执行这个，model，utils是基于这个写的，现在A,B矩阵也写进去了，具体的代价函数根据自己理解写了一版，但是感觉还是有问题，并且权重更新效果不好，模型结果比较差

## autoEnDecoder_Keras.py 和autoEnDecoder_Keras_data.py
用tensorflow实现功能，但是参数不太好改,可以继续研究。autoEnDecoder_Keras_data.py就是加了个数据预处理，之后写进dataprocess中，不同版本

## AutoEncoder.pkl
训练权重，可以参考，现在效果不好

## data
### dataloader.py
解析目录下的Koopman_U和Koopman_Y，并将其做为Train_Dict和Test_Dict输出，返回的是一个字典中有State_Y，State_Init和Input_U三个参数，分别表示Y状态，初始状态，输入


### dataprocess.py
解析dataloader.py输出的字典，将其做为Train_Data,Test_Data,Train_Output,Test_Output四个输出，分别为训练输入，输出，测试输入和输出

### CSTR.m

生成数据matlab程序

## Model

### AutoEncoder.py
编码器解码器的基本结构，输入是20维tensor，十个x，十个u，输出是十个xt+1，并且用到self.matrix_A和self.matrix_A用于参数更新
matrix_A为40 * 40,其是编码器输出shortcut上编码器输入，故为40维tensor，矩阵相乘后需要再将其降维至1维tensor，即tensor(40,1)
matrix_B为10 * 10，其实U内积矩阵，得到输出和上面输出进行shorcut连接
将两者输出喂给解码器去学习


## utils
### lossfunc.py
实现代价计算过程，代价函数是一个字典，维系四个字段，具体loss值还需要学习一下


## example
里面有几个用自编码器结构实现minist数据重建的，可以参考



