library(monocle3)
library(cicero)
library(stringr)
library(Matrix)
library(Seurat)
library(data.table)
library(tidyr)

use_cicero <- function(indata, cellinfo, peakinfo, genome_file, dirpath){
  # make CDS
  input_cds <-  suppressWarnings(new_cell_data_set(indata,
                                                   cell_metadata = cellinfo,
                                                   gene_metadata = peakinfo))
  
  input_cds <- monocle3::detect_genes(input_cds)
  
  #Ensure there are no peaks included with zero reads
  input_cds <- input_cds[Matrix::rowSums(exprs(input_cds)) != 0,]
  
  # use UMAP to reduce dimensionality
  set.seed(0)
  input_cds <- estimate_size_factors(input_cds)
  input_cds <- preprocess_cds(input_cds, method = "LSI")
  input_cds <- reduce_dimension(input_cds, reduction_method = 'UMAP', 
                                preprocess_method = "LSI")
  plot_cells(input_cds)
  
  #access the UMAP coordinates from the input CDS object 
  umap_coords <- reducedDims(input_cds)$UMAP
  
  #run make_cicero_cds
  cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)
  gc()
  
  genome <- fread(genome_file, header = FALSE)#human.hg38.genome,mouse.mm10.genome
  conns <- run_cicero(cicero_cds, genome, sample_num = 100)
  head(conns)
  
  
  conns$Peak2 <- as.character(conns$Peak2)
  
  mask <- !(is.na(conns$coaccess) | (conns$coaccess==0))
  co <- conns$coaccess[mask]
  Peak1 <- conns$Peak1[mask]
  Peak2 <- conns$Peak2[mask]
  
  x <- setNames(seq(nrow(peakinfo)), peakinfo$peak_name)
  id1 <- x[Peak1]
  id2 <- x[Peak2]
  names(id1) <- NULL
  names(id2) <- NULL
  conn_mx <- sparseMatrix(i = id1, j = id2, x = co, dims = c(nrow(peakinfo), nrow(peakinfo)))
  writeMM(conn_mx, file.path(dirpath, "test/PPN.mtx"))
}

make_PPN <-function(dirpath){
  indata <- Matrix::readMM(file.path(dirpath, "X.mtx")) 
  indata@x[indata@x > 0] <- 1
  indata <- t(indata)
  
  # format cell info
  cell_table <- read.csv(file.path(dirpath, "cell_data.csv"))
  colnames(cell_table) <- c('rank','cell_name')
  cell_name <- cell_table$cell_name
  cellinfo <- data.frame(
    cell_name = cell_name, 
    row.names = cell_name
  )
  
  # format peak info
  peak_table <- read.csv(file.path(dirpath, "peak_data.csv"))
  colnames(peak_table) <- c('rank', 'peak_name')
  peak_name <- peak_table$peak_name
  peakinfo=str_split_fixed(peak_name,"-",2)%>%data.frame()
  peakinfo[,c(3,4)] <- str_split_fixed(peakinfo$X2,"-",2)
  peakinfo[,5] <- peak_name
  peakinfo <- peakinfo[,-2]
  names(peakinfo) <- c("chr", "bp1", "bp2", "peak_name")
  row.names(peakinfo) <- peakinfo$peak_name
  peakinfo$id <- seq(1, nrow(peakinfo))
  
  row.names(indata) <- row.names(peakinfo)
  colnames(indata) <- row.names(cellinfo)
  
  use_cicero(indata, cellinfo, peakinfo, "human.hg38.genome", dirpath)
}

make_PPN("data_example/single/")










