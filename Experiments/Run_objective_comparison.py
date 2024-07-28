# Test whether cd can obtain optimal solution at different cases.

from cd_spacer import CD
import time
import pandas as pd
import numpy as np
import causaldag as cd
import functions
import micodag
import temp_micodag


def read_data(graph, n, iter):
    data_path = f"./Data/SyntheticDatasets/graph{graph}"
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
n_samples = [50, 100, 200, 300, 400, 500]
# n_samples = [5000]

# test of CD
for graph_i in range(1, 2):
    for n_sample in n_samples:
        for iter in range(1, 11):
            data, true_dag, true_moral, _ = read_data(graph_i, n_sample, iter)
            N, P = data.shape
            sigma = data.T@data/N
            # true_moral = np.triu(np.ones((P, P)), 1)
            true_moral = true_moral + true_moral.T
            start = time.time()
            est_CD, obj = CD(data, true_moral, MAX_cycles=400, lam= np.sqrt(5*np.log(P) / N))
            end = time.time()
            time_i = end-start
            est_CD_ = np.array([[1 if i != j and est_CD[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
            true_dag_ = cd.DAG.from_amat(true_dag)
            true_cpdag = true_dag_.cpdag().to_amat()
            estimated_dag = cd.DAG.from_amat(est_CD_)
            estimated_cpdag = estimated_dag.cpdag().to_amat()
            SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
            skeleton_estimated, skeleton_true = est_CD_ + est_CD_.T, true_dag + true_dag.T
            SHDs = functions.compute_SHD(skeleton_estimated, skeleton_true, True)
            TPR = np.sum(np.logical_and(est_CD_, true_dag)) / np.sum(true_dag)
            FPR = (np.sum(est_CD_) - np.sum(np.logical_and(est_CD_, true_dag))) / (P * P - np.sum(true_dag))
            result_i = [graph_i, n_sample, iter, SHD_cpdag, obj, TPR, FPR, time_i]
            results_cd.append(result_i)
            print(result_i)
            print(f"TPR: {TPR}; FPR:{FPR}")
result_cd_df = pd.DataFrame(results_cd, columns=['graph', 'n_sample', 'iter', 'd_cpdag', 'obj', 'TPR', 'FPR', 'time'])
# np.sum([-2*np.log(est[i,i]) for i in range(P)]) + np.trace(est@est.T@data.T@data/50) + np.sum(est_)*5*np.log(P)/N  # for debug
# result_cd_df.to_csv('./Results/synthetic_results_CD.csv', index=False)
print(result_cd_df)
print(result_cd_df.mean())


# # # test of micodag
# for graph_i in range(2,3):
#     for n_sample in n_samples:
#         for iter in range(1, 2):
#             data, true_dag, _, true_moral = read_data(graph_i, n_sample, iter)
#             N, P = data.shape
#             # true_moral = true_moral + true_moral.T
#             start = time.time()
#             RGAP, est, run_time, obj = temp_micodag.optimize(data, true_moral, 5*np.log(P) / N)
#             end = time.time()
#             time_i = end-start
#             est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
#             true_dag_ = cd.DAG.from_amat(true_dag)
#             true_cpdag = true_dag_.cpdag().to_amat()
#             estimated_dag = cd.DAG.from_amat(est_)
#             estimated_cpdag = estimated_dag.cpdag().to_amat()
#             SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
#             skeleton_estimated, skeleton_true = est_ + est_.T, true_dag + true_dag.T
#             SHDs = functions.compute_SHD(skeleton_estimated, skeleton_true, True)
#             TPR = np.sum(np.logical_and(est_, true_dag)) / np.sum(true_dag)
#             FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag))) / (P * P - np.sum(true_dag))
#             result_i = [graph_i, n_sample, iter, SHD_cpdag, obj, RGAP, TPR, FPR, time_i]
#             results_micodag.append(result_i)
#             print(result_i)
#             print(f"TPR: {TPR}; FPR:{FPR}")
# result_micodag_df = pd.DataFrame(results_micodag, columns=['graph', 'n_sample', 'iter', 'd_cpdag', 'obj', 'RGAP', 'TPR', 'FPR', 'time'])
# # result_micodag_df.to_csv('./Results/synthetic_results_micodag.csv', index=False)
# print(result_micodag_df)


