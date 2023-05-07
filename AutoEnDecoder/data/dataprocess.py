import numpy as np


def Process_data(Train_Dict,Test_Dict):
    Train_Data = []
    Train_Output = []
    Test_Data = []
    Test_Output = []

    for i in range(len(Train_Dict['State_Y'])):
        x_train = np.ones((len(Train_Dict['State_Init'][i]), 2))
        y_train = np.ones((len(Train_Dict['State_Y'][i]), 1))
        for j in range(len(Train_Dict['State_Y'][i])):
            x_train[j][1] = Train_Dict['Input_U'][i][j]
            x_train[j][0] = Train_Dict['State_Init'][i][j]
            y_train[j][0] = Train_Dict['State_Y'][i][j]
        Train_Data.append(x_train)
        Train_Output.append(y_train)

    for i in range(len(Test_Dict['State_Y'])):
        x_test = np.ones((len(Test_Dict['State_Init'][i]), 2))
        y_test = np.ones((len(Train_Dict['State_Y'][i]), 1))
        for j in range(len(Test_Dict['State_Y'][i])):
            x_test[j][1] = Test_Dict['Input_U'][i][j]
            x_test[j][0] = Test_Dict['State_Init'][i][j]
            y_test[j][0] = Train_Dict['State_Y'][i][j]
        Test_Data.append(x_test)
        Test_Output.append(y_test)

    Train_Data = np.array(Train_Data)
    Test_Data = np.array(Test_Data)
    Train_Output = np.array(Train_Output)
    Test_Output = np.array(Test_Output)
    Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:]))) # 40 * 10 * 2
    Test_Data = Test_Data.reshape((len(Test_Data), np.prod(Test_Data.shape[1:])))
    Test_Output = Test_Output.reshape((len(Test_Output), np.prod(Test_Output.shape[1:])))

    return Train_Data,Test_Data,Train_Output,Test_Output


if __name__ == "__main__":
    from dataloader import Get_data
    Train_Dict,Test_Dict = Get_data(koopman_U = r'koopman_U.csv',koopman_Y = r'koopman_Y.csv')
    Train_Data,Test_Data,Train_Output,Test_Output = Process_data(Train_Dict, Test_Dict)
    print(Train_Data)