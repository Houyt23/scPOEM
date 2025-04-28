import os
import numpy as np
from scipy import sparse
import time
import ray
from sklearn.decomposition import TruncatedSVD
from scipy.io import mmread

#计算其它对第k个的系数
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
    symmetric= True, 
    q= 95., # q: 0-100
    as_sparse= True,
    random_state= 0,
    if_abs= True): 

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
        if if_abs:
            A = abs(A)           
        
        if symmetric: # place in the end
            A = (A + A.T)/2

            
        if q > 0:
            #计算数据沿指定坐标轴的 qth 百分位数。
            A[abs(A) < np.percentile(abs(A), q)] = 0
        #变为对称矩阵。
        
        else:
            A = A.T
        #diag(A) <- 0
        if as_sparse:
            A = sparse.coo_matrix(A)#=    
        return A


def make_pcNet(data, 
          nComp = 5,
          symmetric = True, 
          q = 95., 
          as_sparse = True, 
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
                symmetric = symmetric, 
                q = q, 
                as_sparse = as_sparse, 
                random_state = random_state)    
             
    if ray.is_initialized():
        ray.shutdown()
    duration = time.time() - start_time
    print('execution time of making pcNet: {:.2f} s'.format(duration))
    return net

def deal_abs(net):
    net = np.abs(net)
    return net
    
def deal_percentile(net, q):
    abs_net = np.abs(net)
    net[abs_net  < np.percentile(abs_net, q)] = 0
    return net
    
def deal_symmetric(net): # place in the end
    net = (net + net.transpose())/2
    return net

#基因网络
def make_GRN(dirpath = None,
            rebuild_GRN = False,
            nComp = 5,
            device_count =1,
            if_abs = True,
            if_symmetric = True,
            q = 95.,
            as_sparse = True,
            random_state = 0):
        data = sparse.csr_matrix(mmread(dirpath+"Y.mtx"))
        #定义存储文件
        if dirpath is not None:
            file_name = os.path.join(dirpath, "GGN.npz")
        # load pcnet
        #重新搭建
        if rebuild_GRN:
            print(f'building GRN of genes...')
            #搭建组内网络w
            net = make_pcNet(data, 
                            nComp = nComp, 
                            symmetric=if_symmetric,
                            q = q,
                            as_sparse = as_sparse, 
                            random_state = random_state,
                            device_count = device_count)

            if dirpath is not None:
                os.makedirs(dirpath, exist_ok = True)
                sparse.save_npz(file_name, net)
                
        else:
            print(f'load GGN')
            if dirpath is not None:
                net = sparse.load_npz(file_name)
                net = net.toarray() if sparse.issparse(net) else net #把稀疏矩阵变成数组
    
            if if_abs:
                net = deal_abs(net)
                    
            if if_symmetric:
                net = deal_symmetric(net)

            if q!=0:
                net = deal_percentile(net, q)

            if as_sparse:
                net = sparse.coo_matrix(net)
        return net

    

if __name__ == "__main__":
    dirpath = "data_example/single/"
    make_GRN(dirpath = dirpath,
            rebuild_GRN = True,
            nComp = 5,
            device_count = 1,
            if_abs = False,
            if_symmetric = False,
            q = 95.,
            as_sparse = True,
            random_state = 0)
