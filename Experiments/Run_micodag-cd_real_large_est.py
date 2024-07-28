# from micodagcd import *
from cd_spacer import *
import time
# import closed_form as cf
from sklearn.covariance import graphical_lasso as glasso


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
    return data, graph_, true_moral_

datasets = ['13pathfinder', '15andes', '16diabetes']
n_samples = np.array([545, 1115, 2065])


results = []
# dataset = "17pigs"
for ii in range(3):
    dataset = datasets[ii]
    n_sample = n_samples[ii]
    d_cpdags = []
    times = []
    for iter in range(1, 11):
        data, true_dag, true_moral = read_data(dataset, n_sample, iter)
        N, P = data.shape
        # true_moral = np.triu(np.ones((P, P)), 1)

        # sigma = np.cov(data.T)
        # ests = cf.closed_form(data.values.T, np.log(P)/N)
        # estimated_precision = ests.A
        # estimated_precision[np.abs(estimated_precision) < 0.1] = 0
        # precision_skeleton = np.triu(estimated_precision != 0, 1)
        # estimated_moral = precision_skeleton + precision_skeleton.T
        # print(np.sum(estimated_moral), np.sum(true_moral), np.sum(estimated_moral*true_dag)/np.sum(true_dag))
        #
        # estimated_moral = pd.read_table(f'./Data/RealWorldDatasets/{dataset}/superstructure_glasso_iter_{iter}.txt', sep=',', header=None)
        # estimated_moral = estimated_moral.values
        # print(np.sum(estimated_moral), np.sum(true_moral), np.sum(estimated_moral * true_dag) / np.sum(true_dag))

        estimated_moral = pd.read_table(
            f'/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasets/{dataset}/superstructure_glasso_iter_{iter}.txt',
            sep=',', header=None)
        estimated_moral = estimated_moral.values
        start = time.time()
        est, _ = CD(data, estimated_moral, MAX_cycles=400, lam=np.sqrt(5*np.log(P) / N), tol=0.01)
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
        print(f"TPR: {TPR}; FPR:{FPR}; d_cpdag:{SHD_cpdag}; Time:{end-start}")
        results.append([dataset, iter, SHDs, TPR, FPR, SHD_cpdag, end-start])
    print(np.mean(d_cpdags), np.mean(times))
    results_df = pd.DataFrame(results, columns=['dataset', 'iter', 'SHD', 'TPR', 'FPR', 'd_cpdag', 'Time'])
    print(results_df)
    # results_df.to_csv('micodag-cd_large_est_results.csv', index=False)

