import numpy as np
import pandas as pd
from scipy import sparse
import math
from scipy.spatial import distance
import os
from scipy.io import mmwrite
import argparse


def get_threshold(X, dirpath):
    q_FDR = 0.05

    num = math.ceil(q_FDR * X.shape[0] * (X.shape[0]-1)/2)
    
    temp = distance.cdist(X, X, 'euclidean')

    rows, cols = temp.shape
    m, n = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")  # 生成行、列索引

    mask = m < n
    filtered_m = m[mask].flatten()  # i + m，转换为一维
    filtered_n = n[mask].flatten()        # 保持 n 的索引
    temp_values = temp[mask].flatten()    # 保持对应的距离值

    result = np.column_stack((temp_values, filtered_m, filtered_n))
    result = result[result[:, 0].argsort()]
    result = result[:num, :]  # 只保留最小的 num 个距离

    rows = result[:,1].astype(int)
    cols = result[:,2].astype(int)
    vals = np.exp(-result[:,0])

    sparse_matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    sparse_matrix = sparse_matrix + sparse_matrix.T
    mmwrite(os.path.join(dirpath, "test/eNN_5.mtx"), sparse_matrix)
    return sparse_matrix


def get_eNN(dirpath):
    peak_names = pd.read_csv(os.path.join(dirpath, "peak_data.csv"))
    node_used = np.load(os.path.join(dirpath, "test/node_used.npz"))['arr_0']
    node_used_peak = node_used[node_used<len(peak_names)]
    X = np.load(os.path.join(dirpath, "embedding/node_embeddings.npz"))['arr_0'][len(node_used_peak):,:]
    W_KNN = get_threshold(X, dirpath)
    return W_KNN



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, default="data_example/compare/")
    parser.add_argument("--state1", type=str, default="S1")
    parser.add_argument("--state2", type=str, default="S2")
    args = parser.parse_args()
    get_eNN(os.path.join(args.dirpath, args.state1))
    get_eNN(os.path.join(args.dirpath, args.state2))

    