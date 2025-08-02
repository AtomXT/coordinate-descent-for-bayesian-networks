from src.cd_spacer import *
from collections import deque

import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def topological_sort(adj_matrix):
    n = len(adj_matrix)
    in_degree = [0] * n

    # Step 1: Compute in-degrees
    for j in range(n):
        for i in range(n):
            if adj_matrix[i][j]:
                in_degree[j] += 1

    # Step 2: Initialize queue with nodes of in-degree 0
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in range(n):
            if adj_matrix[u][v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if len(topo_order) != n:
        raise ValueError("The graph is not a DAG (contains a cycle).")

    return topo_order

def read_data(network, n=500, iter=1):
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
    true_moral = pd.read_table(true_moral_path, delimiter=',', dtype=int, header=None)
    graph_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    true_moral_ = [[0] * data.shape[1] for i in range(data.shape[1])]
    for i in range(len(graph)):
        graph_[graph.iloc[i, 0]-1][graph.iloc[i, 1]-1] = 1
    for i in range(len(true_moral)):
        true_moral_[true_moral.iloc[i, 0] - 1][true_moral.iloc[i, 1] - 1] = 1
    graph_, true_moral_ = np.array(graph_), np.array(true_moral_)
    return data, graph_, moral, true_moral_


datasets = ['6cloud']
results = []
np.random.seed(2025)
for dataset in datasets:
    d_cpdags = []
    times = []
    for iter in range(1, 11):
        data, true_dag, moral_lasso, true_moral = read_data(dataset, 500, iter)
        N, P = data.shape

        # randomized reorder
        random_order = np.random.permutation(P)
        # random_order = np.array([i for i in range(P)])

        # Reorder the adjacency matrix
        new_true_dag = true_dag[random_order, :][:, random_order]
        new_true_moral = true_moral[random_order, :][:, random_order]
        new_data = data[list(data.columns[random_order])]

        # start_G2 = np.linalg.cholesky(np.linalg.inv(np.cov(new_data.T))).T
        # model = GraphicalLasso(alpha=0.5)  # alpha controls sparsity (higher = more sparse)
        # model.fit(new_data)
        #
        # # Access precision (inverse covariance) and covariance
        # start_G = np.triu(model.precision_)

        # start_G2[np.abs(start_G2) < 0.3] = 0
        # print(start_G2)

        estimated_moral = pd.read_table(f'{current_dir}/../Data/RealWorldDatasets/{dataset}/superstructure_glasso_iter_{iter}.txt', sep=',', header=None)
        estimated_moral = estimated_moral.values
        new_estimated_moral = estimated_moral[random_order, :][:, random_order]
        # print(np.sum(estimated_moral), np.sum(true_moral), np.sum(estimated_moral * true_dag) / np.sum(true_dag))

        indices = np.where(new_estimated_moral != 0)
        possible_edges_est = tuple(zip(indices[0], indices[1]))
        fixed_gaps_est = set()
        for i in range(P):
            for j in range(P):
                if (i, j) not in possible_edges_est:
                    fixed_gaps_est.add((i, j))
        start = time.time()
        suffstat = partial_correlation_suffstat(new_data)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
        est_dag = gsp(set(range(P)), ci_tester, fixed_gaps=fixed_gaps_est)
        end = time.time()
        times.append(end-start)
        true_dag_ = cd.DAG.from_amat(new_true_dag)
        true_cpdag = true_dag_.cpdag().to_amat()
        B_arcs = est_dag.to_amat()[0]
        ordering = topological_sort(B_arcs)



        estimated_dag = cd.DAG.from_amat(np.array(B_arcs))
        estimated_cpdag = estimated_dag.cpdag().to_amat()
        SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
        d_cpdags.append(SHD_cpdag)
        skeleton_estimated, skeleton_true = skeleton(B_arcs), skeleton(new_true_dag)
        SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
        TPR = np.sum(np.logical_and(B_arcs, new_true_dag)) / np.sum(new_true_dag)
        FPR = (np.sum(B_arcs) - np.sum(np.logical_and(B_arcs, new_true_dag))) / (P * P - np.sum(new_true_dag))
        print(f"TPR: {TPR}; FPR:{FPR}")
        results.append([dataset, iter, SHD_cpdag, end-start, SHDs, TPR, FPR])
    print(np.mean(d_cpdags), np.mean(times))

results_df = pd.DataFrame(results, columns=['dataset', 'iter', 'd_cpdag', 'Time', 'SHDs', 'TPR', 'FPR'])
print(results_df)
# results_df.to_csv(f'/Users/tongxu/Downloads/projects/MICODAG-CD/experiment results/comparison with benchmarks/RealGraph_estimated_superstructure_reorder.csv', index=False)



