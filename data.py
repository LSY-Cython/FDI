from sklearn import preprocessing
import os
import numpy as np

def split_time_series(data,window_size,label,folder):
    series_len = data.shape[0]
    for i in range(series_len//window_size):
        raw_seg = data[i*window_size:(i+1)*window_size,:]
        scaler = preprocessing.MinMaxScaler()
        scaled_seg = scaler.fit_transform(raw_seg)  # 局部片段归一化
        file_name = f"{label}_{i}.npy"
        output_path = os.path.join(folder,file_name)
        np.save(output_path,scaled_seg)

def str_to_array(str_data):
    array_data = list()
    for data in str_data:
        array_data.append([])
        value_data = data[0].split(" ")
        for value in value_data:
            if value != '':
                array_data[-1].append(float(value))
    return np.array(array_data)