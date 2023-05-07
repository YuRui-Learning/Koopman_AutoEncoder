import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from data import dataloader

epoch_time = 1
# 构建自编码器模型
input_data = Input(shape=(2))
encoded = Dense(32, activation='relu')(input_data)
decoded = Dense(1, activation='sigmoid')(encoded)
autoencoder = Model(input_data, decoded)

# 读取数据
Train_Dict , Test_Dict = dataloader.getdata()

Train_Data = []

for i in range(len(Train_Dict['State_Y'])):
    x_train = np.ones((len(Train_Dict['State_Init'][i]), 2))
    y_train = np.ones((len(Train_Dict['State_Y'][i]), 1))
    for j in range(len(Train_Dict['State_Y'][i])):
        x_train[j][1] = Train_Dict['Input_U'][i][j]
        x_train[j][0] = Train_Dict['State_Init'][i][j]
        y_train[j][0] = Train_Dict['State_Y'][i][j]
    Train_Data.append(x_train)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
for epochs in range(epoch_time):
    # for i in range(len(Train_Dict['State_Y'])):
    #     x_train = np.ones((len(Train_Dict['State_Init'][i]),2))
    #     y_train = np.ones((len(Train_Dict['State_Y'][i]),1))
    #     for j in range(len(Train_Dict['State_Y'][i])):
    #         x_train[j][1] = Train_Dict['Input_U'][i][j]
    #         x_train[j][0] = Train_Dict['State_Init'][i][j]
    #         y_train[j][0] = Train_Dict['State_Y'][i][j]
    #         Train_Data.append(x_train)
            # 编译模型并进行训练
        autoencoder.fit(x_train, y_train, epochs=1, batch_size=256, shuffle=True)




# 提取编码器模型并进行预测
encoder = Model(input_data, encoded)
for i in range(len(Test_Dict['State_Y'])):
    x_test = np.ones((len(Test_Dict['State_Init'][i]), 2))
    for j in range(len(Test_Dict['State_Y'][i])):
        x_test[j][1] = Test_Dict['Input_U'][i][j]
        x_test[j][0] = Test_Dict['State_Init'][i][j]
    # 编译模型并进行训练
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)
    # print(encoded_imgs)
    # print(decoded_imgs)

