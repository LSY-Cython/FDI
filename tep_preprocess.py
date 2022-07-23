import pandas as pd
import matplotlib.pyplot as plt
from data import *

def load_normal_data(window_size,dat_file,id):
    normal_data = pd.read_table(dat_file,header=None,sep=" "*3,engine='python').values
    if normal_data.shape[0]==52:
        normal_data = normal_data.T
    process_data = normal_data[:,0:22]  # 22个过程变量
    control_data = normal_data[:,41:52]  # 11个控制变量
    cat_data = np.concatenate((process_data,control_data),axis=1)  # te-(960, 33), tr-(500, 33)
    split_time_series(cat_data,window_size,id,"dataset/TEP_data/Normal/array")
    # for i in range(0,22,1):
    #     plt.plot(process_data[:,i], color="blue", label=f"XMEAS{i+1}")
    #     plt.legend()
    #     plt.savefig(f"dataset/TEP_data/Normal/plot/{id}_XMEAS{i+1}.png")
    #     plt.clf()
    # for j in range(0,11,1):
    #     plt.plot(control_data[:,j], color="blue", label=f"XMV{j+1}")
    #     plt.legend()
    #     plt.savefig(f"dataset/TEP_data/Normal/plot/{id}_XMV{j+1}.png")
    #     plt.clf()

def load_fault_data(window_size,id):
    fault_data = pd.read_table(f"dataset/TEP_data/{id}.dat",header=None).values  # 负号占用空格
    fault_data = str_to_array(fault_data)
    if fault_data.shape[0]==52:
        fault_data = fault_data.T
    process_data = fault_data[160:,0:22]  # 22个过程变量
    control_data = fault_data[160:,41:52]  # 11个控制变量
    cat_data = np.concatenate((process_data,control_data),axis=1)
    split_time_series(cat_data,window_size,id,"dataset/TEP_data/Fault/array")
    for i in range(0,22,1):
        plt.plot(process_data[:,i], color="blue", label=f"XMEAS{i+1}")
        plt.legend()
        plt.savefig(f"dataset/TEP_data/Fault/plot/{id}_XMEAS{i+1}.png")
        plt.clf()
    for j in range(0,11,1):
        plt.plot(control_data[:,j], color="blue", label=f"XMV{j+1}")
        plt.legend()
        plt.savefig(f"dataset/TEP_data/Fault/plot/{id}_XMV{j+1}.png")
        plt.clf()

# if __name__ == "__main__":
#     # load_normal_data(window_size=30,dat_file=f"dataset/TEP_data/d00_te.dat",id=f"d00_te")
#     load_fault_data(window_size=30,id=f"d01_te")