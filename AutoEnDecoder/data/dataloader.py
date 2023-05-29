import pandas as pd
import numpy as np
import random
import os


def Get_data(koopman_U = r'data/koopman_U.csv',koopman_Y = r'data/koopman_Y.csv'):
    '''
    获得两个字典，分别是训练集和测试集，字典中是三个矩阵
    State_Y : 状态矩阵
    State_Init : 初始状态矩阵
    Input_U : 输入矩阵
    '''
    Data_Dict = {'State_Y':[],'State_Init':[],'Input_U':[]}
    Train_Dict = {'State_Y':[],'State_Init':[],'Input_U':[]}
    Test_Dict = {'State_Y':[],'State_Init':[],'Input_U':[]}
    # koopman_U = r'data/koopman_U.csv'
    # koopman_Y = r'data/koopman_Y.csv'
    df_U = pd.read_csv(koopman_U,header=None).T
    df_Y = pd.read_csv(koopman_Y,header=None).T
    init_num = np.array(df_Y.loc[0])
    State_Y = np.array(df_Y.drop(0)).T
    Input_U = np.array(df_U).T


    State_Init = np.ones((len(Input_U.T[0]),len(Input_U[0])))

    for i in range(len(Input_U.T[0])):
        for j in range(len(Input_U[0])):
            State_Init[i][j] = init_num[i]
    Data_Dict['State_Y'] = list(State_Y)
    Data_Dict['State_Init'] = list(State_Init)
    Data_Dict['Input_U'] = list(Input_U)
    nums = random.sample(range(0, len(Input_U.T[0])), len(Input_U.T[0])//5)  # 选取10个元素
    for i in range(len(Input_U.T[0])):
        if i in nums:
            Test_Dict['State_Y'].append(Data_Dict['State_Y'][i])
            Test_Dict['State_Init'].append(Data_Dict['State_Init'][i])
            Test_Dict['Input_U'].append(Data_Dict['Input_U'][i])
        else:
            Train_Dict['State_Y'].append(Data_Dict['State_Y'][i])
            Train_Dict['State_Init'].append(Data_Dict['State_Init'][i])
            Train_Dict['Input_U'].append(Data_Dict['Input_U'][i])
    return Train_Dict , Test_Dict

if __name__ == "__main__":
    Get_data(koopman_U = r'koopman_U.csv',koopman_Y = r'koopman_Y.csv')