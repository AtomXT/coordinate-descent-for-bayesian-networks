'''
Date: 2025/08/09
Author: Tong Xu
Mixed integer convex programming with perspective strengthening + outer approximation.
'''

# import packages
import gurobipy as gp
from gurobipy import GRB
import timeit
import networkx as nx
import random
import numpy as np
import pandas as pd
import time
import os

import causaldag as cd
import scipy
import cvxpy as cp

import src.micodag as mic

from src.functions import *


current_dir = os.path.dirname(os.path.abspath(__file__))

def read_data(network, iter=1, n=500):
    folder_path = os.path.join(current_dir, "../Data/RealWorldDatasetsTXu_smallalpha/")
    data_path = folder_path + f"{network}/data_{network}_n_{n}_iter_{iter}.csv"
    file_path = folder_path + f"{network}"
    graph_name = [i for i in os.listdir(
        file_path) if os.path.isfile(os.path.join(file_path, i)) and 'Sparse_Original_edges' in i][0]
    graph_path = folder_path + network + f"/{graph_name}"
    moral_graph_name = [i for i in os.listdir(
        file_path) if os.path.isfile(os.path.join(file_path, i)) and 'Sparse_Moral_edges' in i][0]
    moral_path = folder_path + network + f"/superstructure_glasso_iter_{iter}.txt"
    true_moral_path = folder_path + network + f"/{moral_graph_name}"
    data, graph = pd.read_csv(data_path, header=None), pd.read_table(graph_path, delimiter=',', dtype=int, header=None)
    moral = pd.read_table(moral_path, delimiter=',', dtype=int, header=None)
    moral = pd.DataFrame(np.where(moral != 0)).T+1
    true_moral = pd.read_table(true_moral_path, delimiter=',', dtype=int, header=None)
    graph_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    true_moral_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    for i in range(len(graph)):
        graph_[graph.iloc[i, 0]-1][graph.iloc[i, 1]-1] = 1
    for i in range(len(true_moral)):
        true_moral_[true_moral.iloc[i, 0] - 1][true_moral.iloc[i, 1] - 1] = 1
    graph_, true_moral_ = np.array(graph_), np.array(true_moral_)
    return data, graph_, moral, true_moral_




if __name__ == '__main__':
    # print(optimization(['9insurance', 'corest']))
    # print(optimization(['10factors', 'true']))
    results = []
    # '1dsep', '2asia', '3bowling', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar'
    # '1dsep', '2asia', '3bowling', '4insuranceSmall','5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar'
    for dataset in ['12hepar']:
        for kk in range(1, 11):
            data, true_dag, moral_, _ = read_data(dataset, iter=kk)
            n, P = data.shape
            gurobi_parameters = {
                'OutputFlag': 1,
                'Threads': 8
            }
            start_i = time.time()
            RGAP, est, _, _, run_time = mic.optimize(data, moral_, np.sqrt(6 * np.log(P) / n), gurobi_params=gurobi_parameters)
            end_i = time.time()
            est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
            true_dag_ = cd.DAG.from_amat(true_dag)
            true_cpdag = true_dag_.cpdag().to_amat()
            estimated_dag = cd.DAG.from_amat(est_)
            estimated_cpdag = estimated_dag.cpdag().to_amat()
            SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
            skeleton_estimated, skeleton_true = skeleton(est_), skeleton(true_dag)
            SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
            TPR = np.sum(np.logical_and(est_, true_dag)) / np.sum(true_dag)
            FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag))) / (P * P - np.sum(true_dag))
            results_i = [dataset, kk, RGAP, SHD_cpdag, SHDs, TPR, FPR, end_i-start_i]
            print([dataset, kk] + list(results_i))
            results.append(results_i)
            df = pd.DataFrame(results, columns=['dataset', 'k', 'RGAP', 'd_cpdag', 'SHDs', 'TPR', 'FPR', 'Time'])
            print(df)
            result_file = os.path.join(current_dir, "../experiment results/comparison with benchmarks/micodag_est_12log(m)n_small_diff_job4.csv")
            df.to_csv(result_file, index=False, header=True)
        #
