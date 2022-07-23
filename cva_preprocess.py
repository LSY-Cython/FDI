import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from data import *

def load_normal_data(window_size):
    normal_data = sio.loadmat("dataset/CVACaseStudy/Training.mat")
    t1_data = normal_data["T1"]  # (10372, 24)
    t2_data = normal_data["T2"]  # (9825, 24)
    t3_data = normal_data["T3"]  # (13200, 24)
    split_time_series(t1_data,window_size,"T1","dataset/CVACaseStudy/Normal/array")
    split_time_series(t2_data,window_size,"T2","dataset/CVACaseStudy/Normal/array")
    split_time_series(t3_data,window_size,"T3","dataset/CVACaseStudy/Normal/array")
    for i in range(24):
        label = f"Variable{i+1}"
        plt.subplot(3,1,1)
        plt.plot(t1_data[:,i],color="blue",label=label)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(t2_data[:,i],color="blue",label=label)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(t3_data[:,i],color="blue",label=label)
        plt.legend()
        plt.savefig(f"dataset/CVACaseStudy/Normal/plot/{label}.png")
        plt.clf()

def load_fault_data(window_size,id):
    fault_data = sio.loadmat(f"dataset/CVACaseStudy/FaultyCase{id}.mat")
    s1_data = fault_data[f"Set{id}_1"]
    s2_data = fault_data[f"Set{id}_2"]
    s3_data = fault_data[f"Set{id}_3"]
    split_time_series(s1_data,window_size,f"Set{id}_1",f"dataset/CVACaseStudy/Fault{id}/array")
    split_time_series(s2_data,window_size,f"Set{id}_2",f"dataset/CVACaseStudy/Fault{id}/array")
    split_time_series(s3_data,window_size,f"Set{id}_3",f"dataset/CVACaseStudy/Fault{id}/array")
    for i in range(24):
        label = f"Variable{i+1}"
        plt.subplot(3,1,1)
        plt.plot(s1_data[:,i],color="blue",label=label)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(s2_data[:,i],color="blue",label=label)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(s3_data[:,i],color="blue",label=label)
        plt.legend()
        plt.savefig(f"dataset/CVACaseStudy/Fault{id}/plot/{label}.png")
        plt.clf()

# if __name__ == "__main__":
#     # load_normal_data(window_size=30)
#     load_fault_data(window_size=30,id=1)