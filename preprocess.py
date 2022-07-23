import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import pickle as pkl

def read_tep_data(dat_file):
    tep_data = pd.read_table(dat_file, header=None).values
    sample_mat = list()
    for sample in tep_data:
        sample_mat.append([])
        value_list = sample[0].split(" ")
        for value in value_list:
            if value != '':
                sample_mat[-1].append(float(value))
    sample_arr = np.array(sample_mat, dtype=np.float32)
    if sample_arr.shape[0]==960 or sample_arr.shape[0]==500:
        pass
    else:
        sample_arr = sample_arr.T  # (960, 52) or (500, 52)
    sensor_data = sample_arr[:, 0:22]  # 22个过程变量
    actuator_data = sample_arr[:, 41:52]  # 11个控制变量
    process_data = np.concatenate((sensor_data, actuator_data), axis=1)  # (960, 33) or (500, 33)
    # for sensorId in range(0,22,1):
    #     plt.plot(sample_arr[:, sensorId], color="blue", label=f"XMEAS{sensorId+1}")
    #     plt.legend()
    #     plt.show()
    # for actuatorId in range(22,33,1):
    #     plt.plot(sample_arr[:, actuatorId], color="red", label=f"XMV{actuatorId-21}")
    #     plt.legend()
    #     plt.show()
    return process_data

def standarlize_tep_data(train_file):
    normal_train = read_tep_data(train_file)
    normal_scaler = preprocessing.MinMaxScaler()
    normal_scaler.fit(normal_train)
    normal_scaled = normal_scaler.transform(normal_train)  # 共960个正常训练样本
    return normal_scaled, normal_scaler

def read_swat_data(pkl_file):
    # 读取速度：pkl>csv>xls/xlsx
    # swat_data = pd.read_csv(csv_file, header=None).iloc[1:, 1:-1].values
    # pkl.dump(np.asarray(swat_data, np.float32), open("dataset/SWaT_data/msl/test_data.pkl", "wb"))
    with open(pkl_file, 'rb') as f:
        swat_data = pkl.loads(f.read())

    # if "normal" in pkl_file:  # 前5-6小时为开机启动过程, 取2015.12.28/16:00:00PM-2015.12.29/16:00:00PM, (86400, 51)
    #     pkl.dump(swat_data[86400:155520], open("dataset/SWaT_data/normal_train.pkl", "wb"))  # (69120, 51)
    #     pkl.dump(swat_data[155520:172800], open("dataset/SWaT_data/normal_valid.pkl", "wb"))  # (17280, 51)
    # if "anomaly" in pkl_file:  # 取2015.12.28/10:00:00AM-2015.12.29/14:28:20AM, (16100, 51)
    #     pkl.dump(swat_data[0:16100], open("dataset/SWaT_data/anomaly_test.pkl", "wb"))  # (16100, 51)

    # with open("dataset/SWaT_data/msl/train_data.pkl", 'rb') as f:
    #     normal_data = pkl.loads(f.read())
    # with open("dataset/SWaT_data/msl/test_data.pkl", 'rb') as f:
    #     anomaly_data = pkl.loads(f.read())
    # for i in range(0, 27, 1):
    #     sensorId = i
    #     plt.subplot(2, 1, 1)
    #     plt.plot(normal_data[:, sensorId], color="blue", label=f"Sensor{sensorId+1}")
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(anomaly_data[:, sensorId], color="red", label=f"Sensor{sensorId+1}")
    #     plt.legend()
    #     plt.show()
    # actuatorId = 2
    # plt.plot(swat_data[:, actuatorId], color="red", label=f"Actuator{actuatorId-21}")
    # plt.legend()
    # plt.show()
    return swat_data
# read_swat_data("dataset/SWaT_data/msl/train_data.pkl")

swatSensorId = np.array([0,1,2,4,7,8,9,10,11,12,14,15,16,17,19,20,22,23,24,25,26])
swatActuatorId = np.array([3,5,6,13,18,21])

def standarlize_swat_data(train_file):
    normal_train = read_swat_data(train_file)
    # print("训练数据规模：", normal_train.shape)
    normal_scaler = preprocessing.MinMaxScaler()
    normal_scaler.fit(normal_train)
    normal_scaled = normal_scaler.transform(normal_train)  # 共86400个正常训练样本
    return normal_scaled, normal_scaler

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, step_size):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size

    def __getitem__(self, index):
        x = self.data[self.step_size*index:self.step_size*index+self.window_size, :]
        return x

    def __len__(self):
        return int((len(self.data)-self.window_size)/self.step_size)+1

class PredictionDataset(Dataset):
    def __init__(self, data, window_size, step_size):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size

    def __getitem__(self, index):
        x = self.data[self.step_size*index:self.step_size*index+self.window_size, :]
        y = self.data[self.step_size*index+self.window_size, :]
        return x, y

    def __len__(self):
        return int((len(self.data)-self.window_size)/self.step_size)

class CorrelationImagesDataset(Dataset):
    def __init__(self, data, window_size, step_size):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size

    def __getitem__(self, index):
        x = self.data[self.step_size*index:self.step_size*index+self.window_size, :]  # (win_size, n_features)
        img = np.dot(x.T, x)/x.shape[0]  # (n_features, n_features)
        return np.reshape(img, (1, img.shape[0], img.shape[1]))

    def __len__(self):
        return int((len(self.data)-self.window_size)/self.step_size)+1

class ParallelDataset(Dataset):
    def __init__(self, data, window_size, step_size):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size

    def __getitem__(self, index):
        x = self.data[self.step_size*index:self.step_size*index+self.window_size, :]
        y = self.data[self.step_size*index+self.window_size, :]
        return x, y

    def __len__(self):
        return int((len(self.data)-self.window_size)/self.step_size)

def create_tep_dataset(train_file, valid_file, test_file, win_size, step_size, batch_size, id=None):
    trainScaled, trainScaler = standarlize_tep_data(train_file)
    validData = read_tep_data(valid_file)
    validScaled = trainScaler.transform(validData)
    testData = read_tep_data(test_file)
    testScaled = trainScaler.transform(testData)
    if id is None:
        trainSet = SlidingWindowDataset(trainScaled, win_size, step_size)
        trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
        validSet = SlidingWindowDataset(validScaled, win_size, step_size)
        validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
        testSet = SlidingWindowDataset(testScaled, win_size, step_size)
        testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    else:
        trainSet = CorrelationImagesDataset(trainScaled, win_size, step_size)
        trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
        validSet = CorrelationImagesDataset(validScaled, win_size, step_size)
        validLoader = DataLoader(dataset=validSet, batch_size=1, shuffle=False)
        testSet = CorrelationImagesDataset(testScaled, win_size, step_size)
        testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    return trainLoader, validLoader, testLoader  # (batch_size, win_size, n_features)

# if __name__ == "__main__":
#     read_tep_data("dataset/TEP_data/d00.dat")
# #     trainLoader, validLoader, testLoader = create_tep_dataset(train_file="dataset/TEP_data/d00_te.dat",
# #                                                               valid_file="dataset/TEP_data/d00.dat",
# #                                                               test_file="dataset/TEP_data/d01_te.dat",
# #                                                               win_size=33,
# #                                                               step_size=10,
# #                                                               batch_size=1)
#     trainLoader, validLoader, testLoader = create_tep_dataset(train_file="dataset/TEP_data/d00_te.dat",
#                                                               valid_file="dataset/TEP_data/d00.dat",
#                                                               test_file="dataset/TEP_data/d01_te.dat",
#                                                               win_size=30,
#                                                               step_size=30,
#                                                               batch_size=1,
#                                                               id="images")

    # 可视化对比确定TEP故障变量
    # normal_data = read_tep_data("dataset/TEP_data/d00_te.dat")
    # fault_data = read_tep_data("dataset/TEP_data/d05_te.dat")
    # for varId in range(0, 33, 1):
    #     plt.subplot(2, 1, 1)
    #     plt.plot(normal_data[:, varId], color="blue", label=f"variable{varId+1}")
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(fault_data[:, varId], color="red", label=f"variable{varId+1}")
    #     plt.legend()
    #     plt.show()

    # # 绘制多元时序相关性特征图
    # i = 0
    # for img in trainLoader:
    #     output = np.asarray(img[0,0,:,:]*255, dtype=np.int32)
    #     cv2.imwrite(f"dataset/TEP_img/normal/normal_{i}.jpg", output)
    #     i += 1
    # j = 0
    # for img in testLoader:
    #     if j>=16:
    #         output = np.asarray(img[0,0,:,:]*255, dtype=np.int32)
    #         cv2.imwrite(f"dataset/TEP_img/anomaly/anomaly_{j}.jpg", output)
    #     j += 1