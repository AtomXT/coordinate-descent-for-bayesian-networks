from causalchamber.datasets import Dataset
import causalchamber
from causalchamber.utils import graph_to_tikz
from causalchamber.ground_truth import latex_name
import sempler.plot
import sempler.utils
from sklearn.covariance import graphical_lasso as glasso
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from src.cd_spacer import *
import ges


def precision(estimate, truth):
    # TP = edges in estimate that are also in truth
    # FP + TP = total edges in estimate
    return np.logical_and(estimate,truth).sum() / estimate.sum()

def recall(estimate, truth):
    # TP = edges in estimate that are also in truth
    # P = total edges in truth
    return np.logical_and(estimate,truth).sum() / truth.sum()

def f1_score(estimate, truth):
    p = precision(estimate, truth)
    r = recall(estimate, truth)
    return 2 * p * r / (p + r)


# Download the dataset and store it, e.g., in the current directory
dataset = Dataset('lt_interventions_standard_v1', root='/Users/tongxu/Downloads/projects/MICODAG-CD/Data/Chamber/', download=False)

dataset.available_experiments()
experiments = [
    "uniform_reference",
    "uniform_red_strong",
    "uniform_green_strong",
    "uniform_blue_strong",
    "uniform_v_c_strong",
    "uniform_t_ir_1_strong",
    "uniform_t_ir_2_strong",
    "uniform_t_ir_3_strong",
    "uniform_t_vis_1_strong",
    "uniform_t_vis_2_strong",
    "uniform_t_vis_3_strong",
    "uniform_pol_1_strong",
    "uniform_pol_2_strong",
    "uniform_v_angle_1_strong",
    "uniform_v_angle_2_strong",
    "uniform_l_11_mid",
    "uniform_l_12_mid",
    "uniform_l_21_mid",
    "uniform_l_22_mid",
    "uniform_l_31_mid",
    "uniform_l_32_mid",

]
variables = ['red', 'green', 'blue', 'current', 'ir_1', 'ir_2', 'ir_3', 'vis_1', 'vis_2', 'vis_3', 'pol_1', 'pol_2',
             'angle_1', 'angle_2', 'l_11', 'l_12', 'l_21', 'l_22', 'l_31', 'l_32']
sub_indicies = [0,1,2,3,4,5,6,10,11,12,13]
subset_vaiables = [variables[i] for i in sub_indicies]
observational_data = dataset.get_experiment(experiments[0]).as_pandas_dataframe()[variables]
N, P = observational_data.shape
true_dag = causalchamber.ground_truth.graph('lt', 'standard').loc[variables, variables].values
true_edges = list(zip(np.where(true_dag == 1)[0]+1,np.where(true_dag == 1)[1]+1))

G = nx.DiGraph(true_edges)
G_moral = nx.moral_graph(G)
moral_edges = list(G_moral.edges())
true_moral = np.zeros((P,P))
for edge in moral_edges:
    true_moral[edge[0]-1, edge[1]-1] = 1
    true_moral[edge[1]-1, edge[0]-1] = 1


results_chamber = []
for n1 in range(1000, 10001, 1000):
    stand_data = (observational_data.values - np.mean(observational_data.values, axis=0))/np.std(observational_data.values, axis=0)
    # glasso_estimates = glasso(np.cov(stand_data[0:n1, :].T), alpha=0.01, max_iter=1000)
    # estimated_pre = glasso_estimates[1]
    # estimated_moral = np.zeros((20, 20))
    # estimated_moral[np.where(np.triu(estimated_pre, 1) != 0)] = 1
    # estimated_moral += estimated_moral.T
    estimated_moral = pd.read_csv(f'../Data/Chamber/chamber_estimated_moral_N{n1}.csv',header=None).values


    # est, obj = CD(observational_data.iloc[0:n1, :], estimated_moral, MAX_cycles=1000, lam=np.sqrt(np.log(n1)/np.sqrt(n1)), tol=0.01)
    start_i = time.time()
    est, obj = CD_order(observational_data.iloc[0:n1, :], estimated_moral, MAX_cycles=1000, lam=np.sqrt(np.log(n1)/np.sqrt(n1)), cholesky=True, tol=1e-5)
    end_i = time.time()
    est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
    estimated_dag = cd.DAG.from_amat(est_)
    estimated_cpdag = estimated_dag.cpdag().to_amat()
    true_dag_ = cd.DAG.from_amat(true_dag)
    true_cpdag = true_dag_.cpdag().to_amat()
    SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
    TPR = np.sum(np.logical_and(est_, true_dag)) / np.sum(true_dag)
    FPR = (np.sum(est_) - np.sum(np.logical_and(est_, true_dag))) / (P * P - np.sum(true_dag))
    results_chamber.append([n1, SHD_cpdag, TPR, FPR, end_i - start_i])
    print(SHD_cpdag, TPR, FPR)
    print(precision(estimated_cpdag[0], true_dag), recall(estimated_cpdag[0], true_dag), f1_score(estimated_cpdag[0], true_dag))
results_chamber_df = pd.DataFrame(results_chamber, columns=['N', 'd_cpdag', 'TPR', 'FPR', 'Time'])
print(results_chamber_df)
print(results_chamber_df.d_cpdag.mean())
# results_chamber_df.to_csv('micodag-cd_chamber_results_reorder1.csv', index=None)


sub_estimated_cpdag = estimated_cpdag[0][sub_indicies, :][:, sub_indicies]
sempler.plot.plot_graph(sub_estimated_cpdag, labels=subset_vaiables)
print(graph_to_tikz(sub_estimated_cpdag, radius=1.7, labels=[latex_name(v) for v in subset_vaiables]))
