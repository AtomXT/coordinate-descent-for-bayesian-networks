###############################################
# Run EqVarDAG-TD on all NID datasets
###############################################

# This script is for comparison between our algorithm and EqVarDAG-TD
# We will use EqVarDAG_TD.R


# import libraries
library(igraph)
library(pcalg)
library(glue)

# import helper functions
source("R_methods/helper_functions.R")
source("R_methods/EqVarDAG_TD.R")


wd = getwd()
dataset.folder <- paste0(wd,"/Data/RealWorldDatasetsTXu_largealpha")
datasets <- list.files(dataset.folder)

#####################################
# Run for each dataset
#####################################

results <- data.frame()

for (dataset in datasets) {
  print(dataset)
  # collect file paths
  for (kk in c(1:10)) {
    data.file = list.files(paste(dataset.folder,dataset,sep="/"), glue("n_500_iter_{kk}"))[1]
    true.graph.file = list.files(paste(dataset.folder,dataset,sep="/"), "Original")
    mgest.file = list.files(glue("{dataset.folder}/{dataset}/"), glue("superstructure_glasso_iter_{kk}.txt"))
    true.moral.file = list.files(paste(dataset.folder,dataset,sep="/"), "Sparse_Moral")
    
    X = as.matrix(read.csv(paste(dataset.folder,dataset, data.file, sep="/"), header=FALSE))
    true.graph = read.table(paste(dataset.folder,dataset,true.graph.file, sep="/"), header=FALSE, sep=",")
    moral.graph = read.table(paste(dataset.folder,dataset,true.moral.file, sep="/"), header=FALSE, sep=",")
    estimated.moral = as.matrix(read.table(paste(dataset.folder,dataset,mgest.file, sep="/"), header=FALSE, sep=","))
    estimated.moral = !estimated.moral

    # generate a graph object from original graph
    nodes = dim(X)[2]
    ori_gg <- make_empty_graph(n = nodes)  
    for (x in c(1:nrow(true.graph))){
      ori_gg <- ori_gg %>% add_edges(c(true.graph[x,1],true.graph[x,2]))
      
    }
    graph_ori = igraph.to.graphNEL(ori_gg)
    # run
    start_time <- Sys.time()
    result <- EqVarDAG_TD(X)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    
    # result analysis
    
    # generate a graph object from estimated graph
    gdag0m = graph_from_adjacency_matrix(result$adj, mode = "directed", weighted = NULL, diag = TRUE, add.colnames = NULL, add.rownames = NA)
    edges = matrix(as.numeric(get.edgelist(gdag0m, names=TRUE)),ncol = 2)
    graph_pred = igraph.to.graphNEL(gdag0m)

    ordering <- as.integer(sub("V", "", as_ids(topo_sort(graph_from_graphnel(pdag2dag(graph_pred)$graph), mode = "out"))))
    write.table(ordering, file = paste0(paste(dataset.folder,dataset,sep="/"), "/td_topo_order_iter", kk,".txt"), row.names = FALSE, col.names = FALSE)

    cpdag_ori <- dag2cpdag(graph_ori)
    cpdag_pred <- dag2cpdag(graph_pred)
    d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
    SHD <- shd(graph_ori, graph_pred)
    SHDs <- shds(graph_ori, graph_pred)
    rates <- compare.Graphs(graph_ori, graph_pred)
    
    result <- list(dataset=dataset,instance=kk, Time=TIME, d_cpdag=d_cpdag, SHD=SHD, SHDs=SHDs, TPR=rates$TPR, FPR=rates$FPR)
    print(result)
    results = rbind(results, result)
  }
}
print(results)

# write the results into a csv file
# write.csv(results, "./experiment results/comparison with benchmarks/TD_RealGraph_est_extra_large_diff.csv",row.names=FALSE)
