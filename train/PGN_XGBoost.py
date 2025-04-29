import numpy as np
import pandas as pd
from scipy.io import mmread,mmwrite
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from xgboost import XGBRegressor
from tqdm import tqdm  
import os
import argparse

def PGN_XGBoost_shapley(X, Y, feature_set = None, gene_name = None):
    g = Y.shape[1]
    p = X.shape[1]
    PGN_net_matrix = sparse.csr_matrix((p, g))
    for i in tqdm(range(g), desc="Training models for each g", ncols=100): 
        gene = gene_name[i]
        if feature_set is not None:
            gene_info = feature_set[feature_set["gene_name"] == gene]
            if gene_info.empty:
                continue
            start, end = gene_info['start_use'].iloc[0], gene_info['end_use'].iloc[0]
            if start==-1:               
                continue
            if (end-start+1)<=5:
                end = start + 5
            X_g = X[:,start:end+1].toarray()
        else:
            start = 0
            end = p-1
            X_g = X
        Y_g = Y[:, i].toarray().flatten() 
        
        xgb_model = XGBRegressor(n_estimators=100, objective='reg:squarederror', n_jobs=-1)
        xgb_model.fit(X_g, Y_g)
        feature_importances = xgb_model.feature_importances_

        PGN_net_matrix[start:end+1,i] = feature_importances
        PGN_net_matrix.eliminate_zeros()

    #PGN_net_matrix = PGN_net_matrix.tocsr()
    
    return PGN_net_matrix

def process_PGN_XGBoost(sparse_matrix, threshold=0.9):

    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = csr_matrix(sparse_matrix)

    rows, cols = sparse_matrix.shape
    result = lil_matrix((rows, cols))
    for col in range(cols):

        col_data = sparse_matrix[:, col].toarray().flatten()
        non_zero_indices = np.nonzero(col_data)[0]
        non_zero_values = col_data[non_zero_indices]

        sorted_indices = np.argsort(non_zero_values)[::-1]
        sorted_values = non_zero_values[sorted_indices]

        cumulative_sum = np.cumsum(sorted_values)
        cutoff_index1 = np.searchsorted(cumulative_sum, threshold, side="right")
        cutoff_index2 = np.where(sorted_values < 0.05)[0]
        cutoff_index2 = cutoff_index2[0]-1 if len(cutoff_index2) > 0 else len(cumulative_sum)-1
        cutoff_index = np.max([cutoff_index1, cutoff_index2])
        selected_indices = sorted_indices[:cutoff_index + 1]

        result[non_zero_indices[selected_indices], col] = col_data[non_zero_indices[selected_indices]]

    return csr_matrix(result)


def make_PGN_XGB(dirpath):
    peak_data = sparse.csr_matrix(mmread(os.path.join(dirpath, "X.mtx")))
    gene_data = sparse.csr_matrix(mmread(os.path.join(dirpath, "Y.mtx")))
    gene_neibor = pd.read_csv(os.path.join(dirpath, "peakuse_100kbp.csv"))
    gene_name = pd.read_csv(os.path.join(dirpath, "gene_data.csv"))['x'].tolist()
    gene_neibor[['start_use', 'end_use']] = gene_neibor[['start_use', 'end_use']].astype(int)

    os.makedirs(os.path.join(dirpath, "test"), exist_ok = True)
    PGN_XGBoost = PGN_XGBoost_shapley(peak_data, gene_data, gene_neibor, gene_name = gene_name)
    #mmwrite(os.path.join(dirpath, "test/PGN_XGB_100kbp.mtx"), PGN_XGBoost)

    PGN_XGBoost_sparse = process_PGN_XGBoost(PGN_XGBoost, threshold=0.9)
    mmwrite(os.path.join(dirpath, "test/PGN_XGB.mtx"), PGN_XGBoost_sparse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, default="data_example/single/")
    args = parser.parse_args()
    make_PGN_XGB(args.dirpath)

    
    
