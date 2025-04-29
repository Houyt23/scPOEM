import os
import numpy as np
from scipy import sparse
import time
import ray
from sklearn.decomposition import TruncatedSVD
from scipy.io import mmread, mmwrite
import argparse


@ray.remote(num_cpus = 1)
def pcCoefficients(X, K, nComp):
    y = X[:, K] 
    Xi = np.delete(X, K, 1)
    #truncated svd
    svd = TruncatedSVD(n_components=nComp)
    svd.fit(Xi)  
    V=svd.components_.T
    s = svd.singular_values_
    score = Xi@V
    beta = V @ np.diag(1/(s**2)[:nComp]) @ (score.T @ y)
    print(f"{K} 完成！")
    return list(beta)

#得到组内网络矩阵
def pcNet(data, # X: cell * gene
    nComp= 5,
    random_state= 0): 

    data = data.toarray() if sparse.issparse(data) else data 
    if nComp < 2 or nComp >= data.shape[1]:
        raise ValueError('nComp should be greater or equal than 2 and lower than the total number of genes') 
    else:
        np.random.seed(random_state)
        n = data.shape[1] # genes  
        
        X_ray = ray.put(data)
        B = np.array(ray.get([pcCoefficients.remote(X_ray, k, nComp) for k in range(n)]))  
            
        A = np.ones((n, n), dtype=float)
        np.fill_diagonal(A, 0)
        for i in range(n):
            A[i, A[i, :]==1] = B[i, :]
        A = sparse.csr_matrix(A)     
        return A


def make_pcNet(data, 
          nComp = 5,
          random_state = 0,
          device_count = 1):
    start_time = time.time()
    if ray.is_initialized():
        ray.shutdown()
    if device_count ==-1:
        device_count = os.cpu_count()
    ray.init(num_cpus = device_count)
    print(f'ray init, using {device_count} CPUs')
    net = pcNet(data, 
                nComp = nComp, 
                random_state = random_state)    
             
    if ray.is_initialized():
        ray.shutdown()
    duration = time.time() - start_time
    print('execution time of making pcNet: {:.2f} s'.format(duration))
    return net


#基因网络
def make_GGN(dirpath,
            rebuild_GRN = False,
            nComp = 5,
            device_count =1,
            random_state = 0):
        data = sparse.csr_matrix(mmread(os.path.join(dirpath, "Y.mtx")))
        if dirpath is not None:
            file_name = os.path.join(dirpath, "test", "GGN.mtx")

        if rebuild_GRN:
            print(f'building GRN of genes...')
            net = make_pcNet(data, 
                            nComp = nComp, 
                            random_state = random_state,
                            device_count = device_count)

            if dirpath is not None:
                os.makedirs(os.path.join(dirpath,"test"), exist_ok = True)
                mmwrite(file_name, net)
                
        else:
            print(f'load GGN')
            if dirpath is not None:
                net = sparse.load_npz(file_name)
                net = net.toarray() if sparse.issparse(net) else net
                net = sparse.csr_matrix(net)
        return net

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, default="data_example/single/")
    parser.add_argument("--count_device", type=int, default=1)
    args = parser.parse_args()
    GGN_net = make_GGN(dirpath = args.dirpath,
                       rebuild_GRN = True,
                       nComp = 5,
                       device_count = args.count_device,
                       random_state = 0)
    print(GGN_net)
