library(pcalg)
library(igraph)
library(glue)


source("/Users/tongxu/Downloads/projects/MICODAG-CD//R_methods/helper_functions.R")

setwd("/Users/tongxu/Downloads/projects/MICODAG-CD")
dataset.folder <- "/Users/tongxu/Downloads/projects/MICODAG-CD/Data/SyntheticDatasets/"
graphs <- c("graph1", "graph2", "graph3")
n_samples <- c(50, 100, 200, 300, 400, 500)



results <- data.frame()

for (dataset_i in c(1:3)) {
  dataset = graphs[dataset_i]
  print(dataset)
  # collect file paths
  for (nn in n_samples){
  for (kk in c(1:10)) {
    data.file = list.files(paste(dataset.folder,dataset,sep="/"), glue("n_{nn}_iter_{kk}"))[1]
    true.graph.file = list.files(paste(dataset.folder,dataset,sep="/"), "DAG")[1]
    # mgest.file = list.files(glue("{dataset.folder}/{dataset}/"), glue("superstructure_glasso_iter_{kk}.txt"))
    true.moral.file = list.files(paste(dataset.folder,dataset,sep="/"), "Moral")
    
    X = as.matrix(read.csv(paste(dataset.folder,dataset, data.file, sep="/"), header=FALSE))
    true.graph = read.table(paste(dataset.folder,dataset,true.graph.file, sep="/"), header=FALSE, sep=",")
    moral.graph = read.table(paste(dataset.folder,dataset,true.moral.file, sep="/"), header=FALSE, sep=",")
    p = dim(X)[2]
    adj_matrix <- matrix(1, nrow = p, ncol = p)
    
    # Populate the adjacency matrix based on the edge list
    for (i in 1:nrow(moral.graph)) {
      adj_matrix[moral.graph[i, 1], moral.graph[i, 2]] <- 0
    }
    
    start_time <- Sys.time()
    score = new("GaussL0penObsScore", data=X, lambda=5*log(dim(X)[2]))
    # score = new("GaussL0penObsScore", data=X)
    ges.fit <- ges(score, fixedGaps = adj_matrix)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    # generate a graph object from original graph
    nodes = dim(X)[2]
    ori_gg <- make_empty_graph(n = nodes)  
    for (x in c(1:nrow(true.graph))){
      ori_gg <- ori_gg %>% add_edges(c(true.graph[x,1],true.graph[x,2]))
      
    }
    graph_ori = as_graphnel(ori_gg)
    
    # result analysis
    
    # generate a graph object from estimated graph
    graph_pred = as(ges.fit$repr,"graphNEL")
    
    cpdag_ori <- dag2cpdag(graph_ori)
    cpdag_pred <- dag2cpdag(graph_pred)
    d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
    # SHD <- shd(graph_ori, graph_pred)
    # SHDs <- shds(graph_ori, graph_pred)
    g1 = wgtMatrix(cpdag_ori, transpose = FALSE)
    g2 = wgtMatrix(cpdag_pred, transpose = FALSE)
    TPR = sum(g2[g1==1])/sum(g1)
    FPR = sum(g2[g1!=1])/(dim(g1)[1]**2 - sum(g2[g1==1]))
    
    
    # compute the score
    
    B = ges.fit$repr$weight.mat()
    D = diag(1/ges.fit$repr$err.var())
    gamma = (diag(p) - B)%*%sqrt(D)
    # obj_i = sum(-2*log(diag(gamma))) + sum(diag(gamma%*%t(gamma)%*%cov(X))) + sum(B!=0)*log(dim(X)[1])/2/dim(X)[1]
    obj_i = sum(-2*log(diag(gamma))) + sum(diag(gamma%*%t(gamma)%*%cov(X))) + 5*sum(B!=0)*log(dim(X)[2])/dim(X)[1]
    
    
    result <- list(graph=dataset_i,n_sample=nn, iter=kk, time=TIME, d_cpdag=d_cpdag, TPR=TPR, FPR=FPR, obj=obj_i)
    print(result)
    results = rbind(results, result)
  }
  }
}
mean(results$d_cpdag)
write.csv(results, "./experiment results/synthetic_results_ges.csv",row.names=FALSE)