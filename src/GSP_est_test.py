import time
import itertools
import gurobipy as gp
from gurobipy import GRB
import timeit
import networkx as nx
import random
import numpy as np
import pandas as pd
import os

import causaldag as cd
import scipy
import cvxpy as cp

from functions import *
from causaldag import rand, partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp


def read_data(network, n=500, iter=1):
    folder_path = "../Data/RealWorldDatasets/"
    data_path = folder_path + f"{network}/data_{network}_n_{n}_iter_{iter}.csv"
    graph_path = folder_path + network + "/Sparse_Original_edges.txt"
    true_moral_path = folder_path + network + "/Sparse_Moral_edges.txt"
    data, graph = pd.read_csv(data_path, header=None), pd.read_table(graph_path, delimiter=',', dtype=int, header=None)
    true_moral = pd.read_table(true_moral_path, delimiter=',', dtype=int, header=None)
    graph_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    true_moral_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    for i in range(len(graph)):
        graph_[graph.iloc[i, 0]-1][graph.iloc[i, 1]-1] = 1
    for i in range(len(true_moral)):
        true_moral_[true_moral.iloc[i, 0] - 1][true_moral.iloc[i, 1] - 1] = 1
    graph_, true_moral_ = np.array(graph_), np.array(true_moral_)

    mgest_name = glob.glob( f"{folder_path}{network}/superstructure_glasso_iter_{kk}.txt")
    mgest = pd.read_table(mgest_name[0], header=None, sep=',')
    return data, graph_, true_moral_, mgest.values


if __name__ == '__main__':
    results = []
    n_samples = [545, 930, 1115, 2065]
    # '1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar'
    # "13pathfinder", "14munin", "15andes", "16diabetes"
    for nn, dataset in enumerate(["13pathfinder", "14munin", "15andes", "16diabetes"]):
        for kk in range(1,11):
            data, true_dag, true_moral, estimate_moral = read_data(dataset, n_samples[nn], kk)
            nnodes, p = data.shape
            nodes = set(range(nnodes))

            indicies = np.where(estimate_moral != 0)
            possible_edges_est = set(zip(indicies[0], indicies[1]))
            # possible_edges_true = tuple(zip(moral.values[:,0]-1, moral.values[:,1]-1))
            all_edges = set(itertools.combinations(range(nnodes),2))
            fixed_gaps_est = all_edges - possible_edges_est
            print(dataset, kk)
            start_i = time.time()
            suffstat = partial_correlation_suffstat(data)
            ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
            est_dag = gsp(nodes, ci_tester, fixed_gaps=fixed_gaps_est)
            end_i = time.time()
            # RGAP, SHD_cpdag, SHDs, TPR, FPR, run_time
            true_dag_ = cd.DAG.from_amat(true_dag)
            true_cpdag = true_dag_.cpdag().to_amat()
            B_arcs = est_dag.to_amat()[0]
            estimated_dag = cd.DAG.from_amat(np.array(B_arcs))
            estimated_cpdag = estimated_dag.cpdag().to_amat()
            SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
            skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(true_dag)
            TPR, FPR = performance(skeleton_estimated, skeleton_true)
            results_i = [dataset, kk, SHD_cpdag, TPR, FPR, end_i-start_i]
            print(results_i)
            results.append(results_i)
        print(pd.DataFrame(results))
        df = pd.DataFrame(results, columns=['dataset', 'k', 'd_cpdag', 'TPR', 'FPR', 'Time'])
        df.to_csv('GSP_large_est_results.csv', index=False)