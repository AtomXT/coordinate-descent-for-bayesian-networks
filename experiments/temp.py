# from micodagcd import *
import numpy as np

from src.cd_spacer import *

import time
import os


current_dir = os.path.dirname(os.path.abspath(__file__))


def read_data(network, n=500, iter=1):
    folder_path = os.path.join(current_dir, "../Data/RealWorldDatasetsTXu_smallalpha1/")
    # folder_path = "/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasets/"
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
    return data, graph_, moral.values, true_moral_


# dataset = "6cloud"
# datasets = ['1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar']
datasets = ['3bowling']
bics = []


for c in range(1, 11):
    for dataset in datasets:
        bic_dataset = []
        for iter in range(1, 11):
            data, true_dag, moral_lasso, true_moral = read_data(dataset, 500, iter)
            N, P = data.shape


            est, _ = CD_order(data, moral_lasso, MAX_cycles=400, lam=np.sqrt(c*np.log(P) / N), cholesky=True)
            sigma_hat = np.cov(data.T)
            theta_hat = est @ est.T
            print(theta_hat)
            bic = -N*(np.log(np.linalg.det(theta_hat)) - np.trace(sigma_hat @ theta_hat))
            + np.sum((np.abs(theta_hat) > 0)) * np.log(N) + 4*np.sum(np.abs(theta_hat) > 0)*np.log(P)
            bic_dataset.append(bic)
        bics.append(np.mean(bic_dataset))
print(bics)
