library(scTenifoldNet)
library(Matrix)
library(data.table)
library(dplyr)


manifold_KNN <- function(dirpath, state1, state2){
  set.seed(0)
  
  KNN_X <- readMM(file.path(dirpath, state1, "test/eNN_5.mtx"))
  KNN_Y <- readMM(file.path(dirpath, state2, "test/eNN_5.mtx"))
  gene_name1 <- read.csv(file.path(dirpath, state1, "gene_data.csv"))$x
  gene_name2 <- read.csv(file.path(dirpath, state2, "gene_data.csv"))$x
  if (!all(mapply(identical, gene_name1, gene_name2))){
    print("Data wrong with different genes!")
    return(0)
  }
  rownames(KNN_X) <- gene_name1
  colnames(KNN_X) <- gene_name1
  rownames(KNN_Y) <- gene_name1
  colnames(KNN_Y) <- gene_name1
  embedding <- manifoldAlignment(KNN_X, KNN_Y, d = 100)
  #embedding_sparse <- readMM("sctenifold/sctenifoldnet_embedding.mtx")
  rownames(embedding) <- c(paste0('X_', gene_name1),paste0('y_', gene_name1))
  # Differential regulation testing
  dR <- dRegulation(manifoldOutput = embedding)
  outputResult <- list()
  outputResult$manifoldAlignment <- embedding
  outputResult$diffRegulation <- dR
  sparse_matrix <- as(embedding, "CsparseMatrix")
  
  outputdir <- file.path(dirpath, "align_embedding")
  if (!dir.exists(outputdir)) {
    dir.create(outputdir, recursive = TRUE)
  }
  writeMM(sparse_matrix, file = file.path(outputdir, "gene_embedding.mtx"))
  write.csv(outputResult$diffRegulation, file=file.path(outputdir, "gene_regulation.csv"))
  write.table(outputResult$diffRegulation$gene[outputResult$diffRegulation$p.adj<0.05], file = file.path(outputdir, "gene_sig.txt"), sep = "\t", quote = FALSE, row.names = FALSE)
}
setwd("/Users/qingzhi/Documents/GitHub/scPOEM")
manifold_KNN("data_example/compare", "S1", "S2")

