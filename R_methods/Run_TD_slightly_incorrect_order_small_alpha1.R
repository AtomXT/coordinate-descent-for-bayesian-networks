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
source("R_methods/EqVarDAG_TD_fixed_order.R")


wd = getwd()
dataset.folder <- paste0(wd,"/Data/RealWorldDatasetsTXu_smallalpha1")
datasets <- c('1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar')

# These indices are randomly generated. We store them to keep the swapped indices the same across methods
swapped_indices <- list(c(1,2), c(5,6), c(1,2), c(12,13), c(10,11), c(4,5), c(7,8), c(16,17), c(2,3), c(7,8), c(2,3), c(28,29))
swap_dict <- setNames(swapped_indices, datasets)
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

    swap_pair <- swap_dict[[dataset]]
    index1 <- swap_pair[1]  
    index2 <- swap_pair[2]
    new_ordering <- 1:nodes 
    new_ordering[c(index1+1, index2+1)] <- c(index2+1, index1+1)

    # run
    start_time <- Sys.time()
    result <- EqVarDAG_TD(X, ordering=new_ordering)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    
    # result analysis
    
    # generate a graph object from estimated graph
    gdag0m = graph_from_adjacency_matrix(result$adj, mode = "directed", weighted = NULL, diag = TRUE, add.colnames = NULL, add.rownames = NA)
    edges = matrix(as.numeric(get.edgelist(gdag0m, names=TRUE)),ncol = 2)
    graph_pred = igraph.to.graphNEL(gdag0m)
    
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
write.csv(results, "./experiment results/cd_vs_regression/TD_slightly_incorrect_ordering_small1_diff.csv",row.names=FALSE)
