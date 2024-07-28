# generate the edge list from .bif file on website: https://www.bnlearn.com/bnrepository/discrete-verylarge.html#andes
library(glue)

folder = "/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasets/"
# filenames <- c('13pathfinder', '14munin', '15andes', '16diabetes', '17pigs', '18link')
data_name = "18link"  ## change this!
data.path = glue("{folder}{data_name}/link.bif.gz") ## and this file name
data = read.bif(data.path)
graph1 = as.igraph(data)
adjlist = as_adj(graph1)
rownames(adjlist) = NULL
colnames(adjlist) = NULL
edge_list <- as_edgelist(graph.adjacency(adjlist))

write.table(edge_list, glue("{folder}{data_name}/Sparse_Original_edges.txt"), row.names=F, col.names=F, sep=',')

# moralize 
moral1 = moral(data)
graph2 = as.igraph(moral1)
adjlist = as_adj(graph2)
rownames(adjlist) = NULL
colnames(adjlist) = NULL
adjlist[lower.tri(adjlist)] = 0
edge_list <- as_edgelist(graph.adjacency(adjlist))
write.table(edge_list, glue("{folder}{data_name}/Sparse_Moral_edges.txt"), row.names=F, col.names=F, sep=',')