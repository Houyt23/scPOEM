library(doParallel)
library(Matrix)
library(glmnet)
library(stringr)
library(dplyr)
library(tictoc)

myic.glmnet <- function (x, y, ...) 
{
  n = length(y)
  model = glmnet(x = x, y = y, alpha = 1, ...)
  coef = coef(model)
  lambda = model$lambda
  df = model$df
  yhat = cbind(1, x) %*% coef
  residuals = (y - yhat)
  mse = colMeans(residuals^2)
  sse = colSums(residuals^2)
  nvar = df + 1
  bic = n * log(mse) + nvar * log(n)
  selected = best.model = which(bic == min(bic))
  ic = c(bic = bic[selected])
  result = list(coefficients = coef[-1, selected], ic = ic, 
                lambda = lambda[selected], nvar = nvar[selected], glmnet = model, 
                fitted.values = yhat[, selected], bic = bic, df = df, call = match.call())
  class(result) = "ic.glmnet"
  return(result)
}

GetPredictionwithCrspPeaksLASSO <- function(X, Y, f=myic.glmnet, neibor_peak, gene_table, count_device){
  
  func.Optimal.lambda.match <-  function(ii){
    gene <- gene_table$gene_name[ii]
    coefficient_extension <- rep(0, ncol(X))
    if (!(gene %in% neibor_peak$gene_name)){
      return(coefficient_extension)
    }
    gene.info <- neibor_peak[neibor_peak$gene_name == gene, ]
    peak.ind <- seq(from = gene.info$start_use+1, to = gene.info$end_use+1)
    
    if(peak.ind[1]==-1){
      coefficient_extension <- Matrix(coefficient_extension, nrow = ncol(X), sparse = TRUE)
      return(coefficient_extension)
    }
    if(length(peak.ind)<=5){
      length_peak = length(peak.ind)
      peak.ind <- seq(from = peak.ind[1], to = min(peak.ind[1]+5, ncol(X)))
    }
    if(sum(Y[,ii])==0){
      coefficient_extension <- Matrix(coefficient_extension, nrow = ncol(X), sparse = TRUE)
      return(coefficient_extension)
    }
    
    icmodel=f(x=X[, peak.ind], y=Y[,ii])
    para.num = icmodel[["nvar"]]-1
    lambda = icmodel[["lambda"]]
    coefficient = icmodel[["coefficients"]]
    if (para.num<5) {
      icmodel=icmodel[["glmnet"]]
      if(max(icmodel[["df"]])<5){
        ind = min(which(icmodel[["df"]]==max(icmodel[["df"]])))
      }
      else{
        ind = min(which(icmodel[["df"]]>=5))
      }
      coefficient=coef(icmodel)[-1, ind]
    }
    
    coefficient_extension[peak.ind] <- coefficient
    coefficient_extension <- Matrix(coefficient_extension, nrow = ncol(X), sparse = TRUE)
    return(coefficient_extension)
  }
  
  
  cl.cores = count_device
  cl = makeCluster(cl.cores)
  registerDoParallel(cl)
  tic()
  result.pre <- foreach(ii=1:ncol(Y),
                        .combine = "cbind",
                        .packages = c("Matrix","glmnet")
  ) %dopar% func.Optimal.lambda.match(ii)
  stopCluster(cl)
  toc()
  colnames(result.pre)=colnames(Y)
  row.names(result.pre)=colnames(X)
  return(result.pre)
}

make_PGN_Lasso <- function(dirpath, count_device){
  X <- Matrix::readMM(file.path(dirpath, "X.mtx")) 
  Y <- Matrix::readMM(file.path(dirpath, "Y.mtx"))
  # format peak info
  
  peak_table <- read.csv(file.path(dirpath, "peak_data.csv"))
  colnames(peak_table) <- c('rank', 'peak_name')
  colnames(X) <- peak_table$peak_name
  gene_table <- read.csv(file.path(dirpath, "gene_data.csv"))
  colnames(gene_table) <- c('rank', 'gene_name')
  colnames(Y) <- gene_table$gene_name
  cell_table <- read.csv(file.path(dirpath, "cell_data.csv"))
  colnames(cell_table) <- c('rank', 'cell_name')
  row.names(X) <- cell_table$cell_name
  row.names(Y) <- cell_table$cell_name
  
  neibor_peak <- read.csv(file.path(dirpath,"peakuse_100kbp.csv"))
  
  result <- GetPredictionwithCrspPeaksLASSO(X, Y, f=myic.glmnet, neibor_peak, gene_table, count_device)
  if (!dir.exists(file.path(dirpath, "test"))) {
    dir.create(file.path(dirpath, "test"))
  }
  writeMM(result, file = file.path(dirpath, "test/PGN_Lasso.mtx"))
}


args <- commandArgs(trailingOnly = TRUE)
dirpath <- NULL
count_device <- NULL

for (i in seq(1, length(args))) {
  if (args[i] == "--dirpath") {
    dirpath <- args[i+1]
  }
  if (args[i] == "--count_device") {
    count_device <- as.integer(args[i+1])
  }
}

if (is.null(dirpath)){
  dirpath <- "data_example/single"
}
if(is.null(count_device)){
  count_device <- 1
}

make_PGN_Lasso(dirpath, count_device)
