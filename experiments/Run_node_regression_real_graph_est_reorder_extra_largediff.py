# from micodagcd import *

from src.cd_spacer import *
from src.node_regression import regression_l0learn
from scipy.sparse.csgraph import floyd_warshall
import time
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

def count_violations_by_adjacency(A_true, A_permuted):
    """
    Count number of edge violations in A_permuted w.r.t. topological structure in A_true.

    Parameters:
    - A_true: (n x n) numpy array, adjacency matrix of the true DAG
    - A_permuted: (n x n) numpy array, same graph but possibly with relabeled nodes

    Returns:
    - Number of edges in A_permuted that violate topological constraints from A_true
    """
    # Compute transitive closure of the true DAG
    # (path[i, j] == True means there's a path from i to j)
    path = floyd_warshall(A_true, directed=True, unweighted=True) < np.inf

    n = A_true.shape[0]
    violations = 0

    for i in range(n):
        for j in range(n):
            if A_permuted[i, j]:  # edge from i -> j in the new graph
                if path[j, i]:  # there's a path from j -> i in the true DAG
                    violations += 1  # so i -> j would violate that

    return violations


def read_data(network, n=500, iter=1):
    folder_path = os.path.join(current_dir, "../Data/RealWorldDatasetsTXu_largealpha/")
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


# dataset = "6cloud"
datasets = ['1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar']
# datasets = ['3bowling']
results = []
np.random.seed(2025)
for dataset in datasets:
    d_cpdags = []
    times = []
    for iter in range(1, 11):
        data, true_dag, moral_lasso, true_moral = read_data(dataset, 500, iter)
        N, P = data.shape

        # feed node regression the same ordering as CD
        ordering = np.array(EqVarDAG_TD_internal(data)['TO'])

        inconsistent_edges = count_violations_by_adjacency(true_dag, true_dag[ordering, :][:, ordering])
        print(f"The estimated ordering causes {inconsistent_edges} edge that is inconsistent with the true topological ordering.")
        start = time.time()
        # est = regression(new_data, ordering, new_estimated_moral, lam=5*np.sqrt(5*np.log(P) / N))
        est = regression_l0learn(data, ordering, moral_lasso.values)
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
results_df.to_csv(f'{current_dir}/../experiment results/comparison with benchmarks/node_regression_results_est_reorder_extra_large_diff.csv', index=False)
# # results_df.to_csv(f'/Users/tongxu/Downloads/projects/MICODAG-CD/experiment results/comparison with benchmarks/RealGraph_estimated_superstructure_reorder.csv', index=False)
