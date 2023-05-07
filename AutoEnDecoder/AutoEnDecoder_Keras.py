import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from data import dataloader
from data import dataprocess

# 读取数据
Train_Dict , Test_Dict = dataloader.Get_data()
# 数据处理，变成 40个 * 10 行 * 2列 数据形式
Train_Data,Test_Data,Train_Output,Test_Output  = dataprocess.Process_data(Train_Dict , Test_Dict)


# 构建自编码器模型
input_data = Input(shape=(20))
encoded = Dense(20, activation='relu')(input_data) # hidden_size
decoded = Dense(10, activation='sigmoid')(encoded) # output_size
autoencoder = Model(input_data, decoded)


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(Train_Data, Train_Output, epochs=400, batch_size=256, shuffle=True)

# 提取编码器模型并进行预测
encoder = Model(input_data, encoded)

# 编译模型并进行训练
encoded_data = encoder.predict(Test_Data)
decoded_data = autoencoder.predict(encoded_data)
print(encoded_data.shape)
print(Test_Output.shape)
# print(decoded_data - Test_Output)
