library(doParallel)
library(Matrix)
library(glmnet)
library(stringr)
library(dplyr)
library(tictoc)

#data loading and binary transforming
setwd("PBMC")
X <- Matrix::readMM("X.mtx") 
Y <- Matrix::readMM("Y.mtx") 
# format peak info

peak_table <- read.csv("peak_data.csv")
colnames(peak_table) <- c('rank', 'peak_name')
colnames(X) <- peak_table$peak_name
gene_table <- read.csv("gene_data.csv")
colnames(gene_table) <- c('rank', 'gene_name')
colnames(Y) <- gene_table$gene_name
cell_table <- read.csv("cell_data.csv")
colnames(cell_table) <- c('rank', 'cell_name')
row.names(X) <- cell_table$cell_name
row.names(Y) <- cell_table$cell_name

neibor_peak <- read.csv("peakuse_chrom.csv")


GetrankwithCrspPeaksLASSO <- function(X, Y, neibor_peak){
  
  func.Optimal.lambda.match <-  function(ii){
    peak.ind <- seq(from = neibor_peak$start_use[ii]+1, to = neibor_peak$end_use[ii]+1)
    rank <- rep(0, ncol(X))
    if(peak.ind[1]==-1){
      rank <- Matrix(rank, nrow = ncol(X), sparse = TRUE)
      return(rank)
    }
    if(sum(Y[,ii])==0){
      rank <- Matrix(rank, nrow = ncol(X), sparse = TRUE)
      return(rank)
    }
    if(length(peak.ind)<=5){
      length_peak = length(peak.ind)
      peak.ind <- seq(from = peak.ind[1], to = peak.ind[1]+5)
    }
    
    model=glmnet(x = X[, peak.ind], y = Y[,ii], alpha = 1)
    coef = coef(model)
    df = model$df
    first_indices <- apply(coef, 1, function(x) {
      first_nonzero <- which(x != 0)
      if (length(first_nonzero) == 0) 0 else first_nonzero[1]+1
    })

    rank[peak.ind] <- first_indices[-1]
    rank <- Matrix(rank, nrow = ncol(X), sparse = TRUE)
    return(rank)
  }
  
  cl.cores = 4
  cl = makeCluster(cl.cores)
  registerDoParallel(cl)
  tic()
  result.rank <- foreach(ii=1:ncol(Y),
                        .combine = "cbind",
                        .packages = c("Matrix","glmnet")
  ) %dopar% func.Optimal.lambda.match(ii)
  stopCluster(cl)
  toc()
  colnames(result.rank)=colnames(Y)
  row.names(result.rank)=colnames(X)
  return(result.rank)
}

result_rank <- GetrankwithCrspPeaksLASSO(X, Y, neibor_peak)
writeMM(result_rank, file = "test/rank.mtx")