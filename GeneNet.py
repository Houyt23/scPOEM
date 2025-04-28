import os
from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
from scipy import sparse
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ray
import torch
from sklearn.decomposition import TruncatedSVD

n_cpus = os.cpu_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    #U, s, VT = svd(Xi, full_matrices=False) 
    #print ('U:', U.shape, 's:', s.shape, 'VT:', VT.shape)
    #只选取签nComp个主成分
    #V = VT[:nComp, :].T
    #print('V:', V.shape)
    score = Xi@V
    beta = V @ np.diag(1/(s**2)[:nComp]) @ (score.T @ y)
    print(f"{K} 完成！")
    return list(beta)

#计算其它对第k个的系数
@ray.remote(num_gpus = 1)
def pcCoefficients_gpu(X, K, nComp):
    id = os.environ["CUDA_VISIBLE_DEVICES"]
    #device = torch.device(f"cuda:{id}")
    X = X.cuda()
    y = X[:,K]
    Xi = torch.cat((X[:, 0:K], X[:, K+1:]),1)
    U, s, VT = torch.linalg.svd(Xi, full_matrices=False) 
    #print ('U:', U.shape, 's:', s.shape, 'VT:', VT.shape)
    #只选取签nComp个主成分
    V = VT[:nComp, :].T
    #print('V:', V.shape)
    score = Xi@V
    beta = V @ torch.diag(1/(s**2)[:nComp]) @ (score.T @ y)
    beta = beta.cpu()
    beta = beta.numpy()
    print(f"{id} {K} 完成！")
    return list(beta)  

#得到组内网络矩阵
def pcNet(X, # X: cell * gene
    nComp: int = 3, 
    scale: bool = True, 
    symmetric: bool = True, 
    q: float = 95., # q: 0-100
    as_sparse: bool = True,
    random_state: int = 0,
    if_gpu: bool = True,
    if_abs: bool = True): 

    X = X.toarray() if sparse.issparse(X) else X #把稀疏矩阵变成数组
    if nComp < 2 or nComp >= X.shape[1]:
        raise ValueError('nComp should be greater or equal than 2 and lower than the total number of genes') 
    else:
        np.random.seed(random_state)
        n = X.shape[1] # genes  
        if if_gpu:  
            X = torch.from_numpy(X).float()
            X_ray = ray.put(X)
            B = np.array(ray.get([pcCoefficients_gpu.remote(X_ray, k, nComp) for k in range(n)])) 
        else:
            X_ray = ray.put(X)
            B = np.array(ray.get([pcCoefficients.remote(X_ray, k, nComp) for k in range(n)])) 
            #B = np.array([pcCoefficients(X, k, nComp) for k in range(n)])   
            
        A = np.ones((n, n), dtype=float)
        np.fill_diagonal(A, 0)
        for i in range(n):
            A[i, A[i, :]==1] = B[i, :]     
        if if_abs:
            A = abs(A)           
        if scale:
            absA = abs(A)
            A = A / np.max(absA)
        
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


def make_pcNet(X, 
          nComp: int = 3, 
          scale: bool = True, 
          symmetric: bool = True, 
          q: float = 95., 
          as_sparse: bool = True, 
          random_state: int = 0,
          if_gpu: bool = False,
          device_count: int = 1,
          **kwargs):
    start_time = time.time()
    if ray.is_initialized():
        ray.shutdown()
        
    if if_gpu:
        ray.init(num_gpus = device_count)
        print(f'ray init, using {device_count} GPUs')
        net = pcNet(X, 
                    nComp = nComp, 
                    scale = scale, 
                    symmetric = symmetric, 
                    q = q, 
                    as_sparse = as_sparse, 
                    random_state = random_state, 
                    if_gpu = True)
    else:
        ray.init(num_cpus = device_count)
        print(f'ray init, using {device_count} CPUs')
        net = pcNet(X, 
                    nComp = nComp, 
                    scale = scale, 
                    symmetric = symmetric, 
                    q = q, 
                    as_sparse = as_sparse, 
                    random_state = random_state, 
                    if_gpu = False)    
             
    if ray.is_initialized():
        ray.shutdown()
    duration = time.time() - start_time
    print('execution time of making pcNet: {:.2f} s'.format(duration))
    return net

#基因网络
class GRN:
    def __init__(self,
                 data: anndata.AnnData = None,
                 GRN_file_dir: Union[str, PathLike] = None,
                 rebuild_GRN: bool = False,
                 nComp: int = 5,
                 if_gpu: bool = False,
                 device_count: int =1,
                 if_abs: bool = True,
                 if_scale: bool = True,
                 if_symmetric: bool = True,
                 q: float = 95.,
                 as_sparse: bool = True,
                 random_state: int = 0,
                 var_name: str = "gene_name", 
                 **kwargs):
        self.kws = kwargs
        #定义存储文件
        if GRN_file_dir is not None:
            self._pc_net_file_name = (Path(GRN_file_dir) / Path(f"GRN.npz"))
        # load pcnet
        #重新搭建
        if rebuild_GRN:
            print(f'building GRN of genes...')
            #搭建组内网络w
            self._net = make_pcNet(data.X, 
                                   nComp = nComp, 
                                   scale=if_scale,
                                   symmetric=if_symmetric,
                                   q = q,
                                   as_sparse = as_sparse, 
                                   random_state = random_state,
                                   if_gpu = if_gpu,
                                   device_count = device_count, 
                                   **kwargs)
            #基因或峰名称
            self._gene_names = data.var_names.copy(deep=True)
            # if verbose:
            #     print(f'GRN of {name} has been built')
            #存储
            if GRN_file_dir is not None:
                os.makedirs(GRN_file_dir, exist_ok = True)
                sparse.save_npz(self._pc_net_file_name, self._net)
                self._gene_names.to_frame(name=var_name).to_csv(Path(GRN_file_dir) / Path(f"GRN_gene_name.tsv"),
                                                                   sep='\t')
        else:
            print(f'load GRN')
            if GRN_file_dir is not None:
                self._gene_names = pd.Index(pd.read_csv(Path(GRN_file_dir) / Path(f"GRN_gene_name.tsv"),
                                                        sep='\t')[var_name])
                self._net = sparse.load_npz(self._pc_net_file_name)
                self._net = self._net.toarray() if sparse.issparse(self._net) else self._net #把稀疏矩阵变成数组
                if if_abs:
                    self.deal_abs()
                

                if if_symmetric:
                    self.deal_symmetric()

                if if_scale:
                    self.deal_scale()

                if q!=0:
                    self.deal_percentile(q)

                

                self._net = sparse.coo_matrix(self._net)
                sns.boxplot(data=self._net.data)
                # 设置图表标题和标签
                plt.title('Gene')
                plt.xlabel('Data')
                # 显示图表
                plt.show()

    def deal_abs(self):
        self._net = np.abs(self._net)

    def deal_scale(self):
        max_num = np.max(self._net.data)
        self._net = self._net/max_num
    
    def deal_percentile(self, q):
        abs_net = np.abs(self._net)
        self._net[abs_net  < np.percentile(abs_net, q)] = 0
        #np.percentile(abs_net, q) 0.3 0.6
    

    def deal_symmetric(self): # place in the end
        self._net = (self._net + self._net.transpose())/2
