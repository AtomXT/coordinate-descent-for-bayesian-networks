###############################################
# Run ges algorithm on chamber data with true moral graph
###############################################

# This script is for comparison between our algorithm and ges algorithm
# We will use pcalg

library(pcalg)
library(igraph)
library(glue)

source("/Users/tongxu/Downloads/projects/MICODAG-CD/R_methods/helper_functions.R")

setwd("/Users/tongxu/Downloads/projects/MICODAG-CD")
X <- read.csv('chamber_data1.csv')
true_moral <- as.matrix(read.csv('./Data/Chamber/chamber_true_moral.csv', header = FALSE))
true_dag <- as.matrix(read.csv('./Data/Chamber/chamber_true_dag.csv', header=FALSE))


# only consider a subset of variables
# indices = c(0,1,2,3,4,5,6,10,11,12,13)
# true_dag = true_dag[indices,indices]
# X = X[, indices]

#####################################
# Run for each dataset
#####################################

results <- data.frame()

for (n1 in 1:10 * 1000) {
  estimated_moral <- as.matrix(read.csv(glue('./Data/Chamber/chamber_estimated_moral_N{n1}.csv'), header=FALSE))
  # estimated_moral <- estimated_moral[indices, indices] # only consider a subset of variables
  start_time <- Sys.time()
  # score = new("GaussL0penObsScore", data=X, lambda=2)
  score = new("GaussL0penObsScore", data=X[1:n1,])
  # ges.fit <- ges(score, fixedGaps = 1-estimated_moral)
  ges.fit <- ges(score)
  end_time <- Sys.time()
  TIME <- as.numeric(end_time - start_time, units="secs")
  # generate a graph object from original graph
  nodes = dim(X)[2]
  
  graph_ori = as_graphnel(graph_from_adjacency_matrix(true_dag))
  
  # result analysis
  
  # generate a graph object from estimated graph
  graph_pred = as(ges.fit$essgraph,"graphNEL")
  
  cpdag_ori <- dag2cpdag(graph_ori)
  cpdag_pred <- dag2cpdag(graph_pred)
  d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
  # SHD <- shd(graph_ori, graph_pred)
  # SHDs <- shds(graph_ori, graph_pred)
  # g1 = wgtMatrix(cpdag_ori, transpose = FALSE)
  # g2 = wgtMatrix(cpdag_pred, transpose = FALSE)
  # TPR = sum(g2[g1==1])/sum(g1)
  # FPR = sum(g2[g1!=1])/(dim(g1)[1]**2 - sum(g2[g1==1]))
  g1 <- wgtMatrix(graph_pred, transpose = FALSE)
  TPR = sum(g1 * true_dag) / sum(true_dag)
  FPR = (sum(g1) - sum(g1 * true_dag)) / (20*20-sum(true_dag))
  
  result <- list(N=n1, Time=TIME, d_cpdag=d_cpdag, TPR=TPR, FPR=FPR)
  print(result)
  results = rbind(results, result)
}
results

mean(results$d_cpdag)
# write.csv(results, "ges_chamber_est_results.csv",row.names=FALSE)
# write.csv(as.matrix(get.adjacency(graph_from_graphnel(cpdag_ori))), 'ges_estimated_moral_cpdag_n10000.csv', row.names=FALSE)
