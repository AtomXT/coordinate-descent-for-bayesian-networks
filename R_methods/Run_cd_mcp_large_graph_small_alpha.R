library(igraph)
library(sparsebn)
library(glue)
library(dplyr)
library(pcalg)


wd = getwd()
dataset.folder <- paste0(wd,"/Data/RealWorldDatasetsTXu_smallalpha")
datasets <- c('13pathfinder', '15andes', '16diabetes')
nsamples <- c('13pathfinder'=5450, '15andes'=11150, '16diabetes'=20650)
results <- data.frame()

for (dataset in datasets) {
  for (kk in c(1:10)) {
    data.file = list.files(paste(dataset.folder,dataset,sep="/"), glue("n_{nsamples[dataset]}_iter_{kk}"))[1]
    true.graph.file = list.files(paste(dataset.folder,dataset,sep="/"), "Original")
    mgest.file = list.files(glue("{dataset.folder}/{dataset}/"), glue("superstructure_glasso_iter_{kk}.txt"))
    true.moral.file = list.files(paste(dataset.folder,dataset,sep="/"), "Sparse_Moral")
    
    X = as.matrix(read.csv(paste(dataset.folder,dataset, data.file, sep="/"), header=FALSE))
    true.graph = read.table(paste(dataset.folder,dataset,true.graph.file, sep="/"), header=FALSE, sep=",")
    moral.graph = read.table(paste(dataset.folder,dataset,true.moral.file, sep="/"), header=FALSE, sep=",")
    estimated.moral = as.matrix(read.table(paste(dataset.folder,dataset,mgest.file, sep="/"), header=FALSE, sep=","))
    
    estimated.moral = as.data.frame(which(estimated.moral==0,arr.ind = T), col.names=NULL)
    colnames(estimated.moral) = c('V1', 'V2')
    blacklist_estimated <- as.matrix(estimated.moral)
    
    dat <- sparsebnData(X, type = "c", levels = NULL, ivn = NULL)
    start_time <- Sys.time()
    est = estimate.dag(dat, blacklist = blacklist_estimated)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    adj = get.adjacency.matrix(est[[7]])
    # write.table(as.matrix(adj), glue("/Users/tongxu/Downloads/projects/MICODAG-CD/Results/SOTA/CD/Estimations/estimated_moral/{network}_iter_{iter}.txt"), row.names = F, col.names = F)
    gdag0m = graph_from_adjacency_matrix(adj, mode = "directed", weighted = NULL, diag = TRUE, add.colnames = NULL, add.rownames = NA)
    graph_pred = as_graphnel(gdag0m)
    
    true.graph_adj <- matrix(0, nrow = ncol(X), ncol = ncol(X))
    for (x in c(1:nrow(true.graph))){
      true.graph_adj[true.graph[x,1],true.graph[x,2]] = 1
    }
    ori_gg = graph_from_adjacency_matrix(true.graph_adj)
    graph_ori = as_graphnel(ori_gg)
    cpdag_ori <- dag2cpdag(graph_ori)
    cpdag_pred <- dag2cpdag(graph_pred)
    d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
    result <- list(dataset=dataset,k=kk, Time=TIME, d_cpdag=d_cpdag)
    results <- rbind(results, result)
    results = as.data.frame(results)
    print(results)
    write.csv(results, "./experiment results/comparison on large graphs/ccdr_mcp_large_graph_small_diff.csv",row.names=FALSE)
  }
}

