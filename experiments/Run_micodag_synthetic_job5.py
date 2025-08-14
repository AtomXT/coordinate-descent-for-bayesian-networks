# Test whether cd can obtain optimal solution at different cases.

from src.cd_spacer import CD, CD_order
import time
import pandas as pd
import numpy as np
import causaldag as cd
from src import functions
import src.micodag as mic
import os

# import temp_micodag
current_dir = os.path.dirname(os.path.abspath(__file__))


def read_data(graph, n, iter):
    data_path = f"{current_dir}/../Data/SyntheticDatasets_obj/graph{graph}"
    graph = pd.read_table(f"{data_path}/DAG.txt", delimiter=',', dtype=int, header=None)
    data = pd.read_csv(f"{data_path}/data_n_{n}_iter_{iter}.csv", header=None)
    true_moral = pd.read_table(f"{data_path}/Moral_DAG.txt", delimiter=',', dtype=int, header=None)
    graph_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    true_moral_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    for i in range(len(graph)):
        graph_[graph.iloc[i, 0] - 1][graph.iloc[i, 1] - 1] = 1
    for i in range(len(true_moral)):
        true_moral_[true_moral.iloc[i, 0] - 1][true_moral.iloc[i, 1] - 1] = 1
    graph_, true_moral_ = np.array(graph_), np.array(true_moral_)
    return data, graph_, true_moral_, true_moral


results_cd = []
results_micodag = []
n_samples = [1600, 3200]


# # test of micodag
for graph_i in range(3,4):
    for n_sample in n_samples:
        for iter in range(1, 11):
            data, true_dag, _, true_moral = read_data(graph_i, n_sample, iter)
            N, P = data.shape
            # true_moral = true_moral + true_moral.T
            gurobi_parameters = {
                'OutputFlag': 1,
                'Threads': 8,
                'TimeLimit': 500
            }
            start = time.time()
            RGAP, est, _, obj, run_time = mic.optimize(data, true_moral, 5*np.log(P) / N, gurobi_params=gurobi_parameters)
            end = time.time()
            time_i = end-start
            est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
            true_dag_ = cd.DAG.from_amat(true_dag)
            true_cpdag = true_dag_.cpdag().to_amat()
            estimated_dag = cd.DAG.from_amat(est_)
            estimated_cpdag = estimated_dag.cpdag().to_amat()
            SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
            skeleton_estimated, skeleton_true = est_ + est_.T, true_dag + true_dag.T
            SHDs = functions.compute_SHD(skeleton_estimated, skeleton_true, True)
            TPR = np.sum(np.logical_and(est_, true_dag)) / np.sum(true_dag)
            FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag))) / (P * P - np.sum(true_dag))
            result_i = [graph_i, n_sample, iter, SHD_cpdag, obj, RGAP, TPR, FPR, time_i]
            results_micodag.append(result_i)
            print(result_i)
            print(f"TPR: {TPR}; FPR:{FPR}")
            result_micodag_df = pd.DataFrame(results_micodag, columns=['graph', 'n_sample', 'iter', 'd_cpdag', 'obj', 'RGAP', 'TPR', 'FPR', 'time'])
            result_micodag_df.to_csv('./experiment results/synthetic_results_micodag_small_diff_part5.csv', index=False,header=True)
            print(result_micodag_df)


