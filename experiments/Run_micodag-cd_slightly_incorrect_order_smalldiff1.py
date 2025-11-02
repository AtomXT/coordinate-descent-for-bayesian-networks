# from micodagcd import *

from src.cd_spacer import *

import time
import os


current_dir = os.path.dirname(os.path.abspath(__file__))


def read_data(network, n=500, iter=1):
    folder_path = os.path.join(current_dir, "../Data/RealWorldDatasetsTXu_smallalpha1/")
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
    true_moral = pd.read_table(true_moral_path, delimiter=',', dtype=int, header=None)
    graph_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    true_moral_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    for i in range(len(graph)):
        graph_[graph.iloc[i, 0]-1][graph.iloc[i, 1]-1] = 1
    for i in range(len(true_moral)):
        true_moral_[true_moral.iloc[i, 0] - 1][true_moral.iloc[i, 1] - 1] = 1
    graph_, true_moral_ = np.array(graph_), np.array(true_moral_)
    return data, graph_, moral, true_moral_


# n_variables = [6, 8, 9, 15, 14, 16, 18, 20, 27, 27, 56, 70]
# swapped_indices = []
# for dataset in n_variables:
#     swapped_indices.append(random.sample(range(dataset), 2))
# print(swapped_indices)
# swapped_indices = [[3, 4], 
# [4, 0], [1, 7], [6, 2], [5, 11], [12, 15], [6, 0], [9, 19], [7, 15], [19, 3], [13, 16], [32, 12]]

swapped_indices = [[1,2],[5,6],[1,2],[12,13],[10,11],[4,5],[7,8],[16,17],[2,3],[7,8],[2,3],[28,29]]

datasets = ['1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar']
# datasets = ['6cloud']
results = []
for data_i, dataset in enumerate(datasets):
    d_cpdags = []
    times = []
    index_1, index_2 = swapped_indices[data_i]
    for iter in range(1, 11):
        data, true_dag, estimated_moral, true_moral = read_data(dataset, 500, iter)
        N, P = data.shape
        new_ordering = np.arange(P)
        new_ordering[index_1], new_ordering[index_2] = index_2, index_1
        estimated_moral = estimated_moral.values
        # print(np.sum(estimated_moral), np.sum(true_moral), np.sum(estimated_moral * true_dag) / np.sum(true_dag))

        start = time.time()
        est, _ = CD_order(data, estimated_moral, ordering=new_ordering, MAX_cycles=400, lam=np.sqrt(5*np.log(P) / N), cholesky=True)
        end = time.time()
        times.append(end-start)
        est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
        true_dag_ = cd.DAG.from_amat(true_dag)
        true_cpdag = true_dag_.cpdag().to_amat()
        estimated_dag = cd.DAG.from_amat(est_)
        estimated_cpdag = estimated_dag.cpdag().to_amat()
        SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
        d_cpdags.append(SHD_cpdag)
        skeleton_estimated, skeleton_true = skeleton(est_), skeleton(true_dag)
        SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
        TPR = np.sum(np.logical_and(est_, true_dag)) / np.sum(true_dag)
        FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag))) / (P * P - np.sum(true_dag))
        print(f"TPR: {TPR}; FPR:{FPR}")
        results.append([dataset, iter, SHD_cpdag, end-start, SHDs, TPR, FPR])
    print(np.mean(d_cpdags), np.mean(times))

results_df = pd.DataFrame(results, columns=['dataset', 'iter', 'd_cpdag', 'Time', 'SHDs', 'TPR', 'FPR'])
print(results_df)
print(results_df.describe())
results_df.to_csv('./experiment results/cd_vs_regression/CD_slightly_incoorect_order_small_diff1.csv', index=False, header=True)
