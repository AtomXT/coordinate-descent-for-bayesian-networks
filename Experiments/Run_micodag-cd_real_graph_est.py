# from micodagcd import *
import pandas as pd

from cd_spacer import *
import time

from sklearn.covariance import graphical_lasso as glasso


def read_data(network, n=500, iter=1):
    folder_path = "/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasets/"
    data_path = folder_path + f"{network}/data_{network}_n_{n}_iter_{iter}.csv"
    graph_path = folder_path + network + "/Sparse_Original_edges.txt"
    moral_path = folder_path + network + f"/superstructure_glasso_iter_{iter}.txt"
    true_moral_path = folder_path + network + "/Sparse_Moral_edges.txt"
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
# datasets = ['1dsep', '2asia']
results = []
for dataset in datasets:
    d_cpdags = []
    times = []
    for iter in range(1, 11):
        data, true_dag, moral_lasso, true_moral = read_data(dataset, 500, iter)
        N, P = data.shape

        estimated_moral = pd.read_table(f'/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasets/{dataset}/superstructure_glasso_iter_{iter}.txt', sep=',', header=None)
        estimated_moral = estimated_moral.values
        # print(np.sum(estimated_moral), np.sum(true_moral), np.sum(estimated_moral * true_dag) / np.sum(true_dag))

        start = time.time()
        est, _ = CD(data, estimated_moral, MAX_cycles=400, lam=np.sqrt(5*np.log(P) / N))
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
        results.append([dataset, iter, SHD_cpdag, end-start, TPR, FPR])
    print(np.mean(d_cpdags), np.mean(times))

results_df = pd.DataFrame(results, columns=['dataset', 'iter', 'd_cpdag', 'Time', 'TPR', 'FPR'])
print(results_df)
results_df.to_csv(f'/Users/tongxu/Downloads/projects/MICODAG-CD/experiment results/RealGraph_estimated_superstructure_0.001_12.csv')
