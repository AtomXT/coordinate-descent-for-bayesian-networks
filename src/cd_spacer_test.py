from collections import deque
import numpy as np
import pandas as pd
import causaldag as cd
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import ges


def EqVarDAG_TD_internal(X):
    n, p = X.shape
    done = [p]  # Initialize with an invalid index (p is out of bounds)
    S = np.cov(X, rowvar=False)  # Compute covariance matrix
    Sinv = np.linalg.inv(S)  # Compute inverse of the covariance matrix

    for i in range(p):
        # Exclude already selected variables
        remaining = [j for j in range(p) if j not in done]
        # Submatrix of Sinv for remaining variables
        Sinv_sub = Sinv[np.ix_(remaining, remaining)]
        # Compute diagonal of the inverse of the submatrix
        diag_inv = np.diag(np.linalg.inv(Sinv_sub))
        # Find the variable with the minimum diagonal value
        v = remaining[np.argmin(diag_inv)]
        done.append(v)

    # Remove the initial invalid index (p) and return the topological order
    return {'TO': done[1:], 'support': None}


def read_data(network, n=500, iter=1):
    folder_path = "../Data/RealWorldDatasets_ID/"
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

def cycle(G, i, j):
    """
    Check whether a DAG G remains acyclic if an edge i->j is added.
    Return True if it is no longer a DAG.

    Examples
    --------
    Consider a DAG defined as:
    dag = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    print(cycle(dag, 3, 2))
    """
    P = len(G)
    C = [0] * P
    C[i] = 1
    Q = deque([i])
    while Q:
        u = Q.popleft()
        parent_u = [ii for ii in range(P) if G[ii, u] != 0]
        for v in parent_u:
            if v == j:
                return True
            else:
                if C[v] == 0:
                    C[v] = 1
                    Q.append(v)
    return False


def g(Gamma, sigma_hat, u, v, lam):
    ans = sum([Gamma[u, v]*Gamma[i, v]*sigma_hat[i, u] for i in range(len(Gamma))])
    ans += Gamma[v, v]*Gamma[u, v]*sigma_hat[u, v]
    ans += sum([Gamma[k, v] * Gamma[u, v] * sigma_hat[u, k] for k in range(len(Gamma)) if k != u and k != v])
    if Gamma[u, v] != 0:
        ans += lam*lam
    return ans


def objective(Gamma, sigma_hat, lam):
    P = len(Gamma)
    obj = sum([-2*np.log(Gamma[i, i]) for i in range(P)])
    obj += np.trace(Gamma@Gamma.T@sigma_hat)
    obj += lam*lam*(np.count_nonzero(Gamma)-P)
    return obj


def gamma_hat(Gamma, sigma_hat, u, v, lam):
    # numerator = sum([Gamma[i, v]*(sigma_hat[i, u]+sigma_hat[u, i]) for i in range(len(Gamma)) if i != u])
    numerator = sum([2 * Gamma[i, v] * sigma_hat[i, u] for i in range(len(Gamma)) if i != u])
    # return -numerator/sigma_hat[u, u]/2 if g(gamma, sigma_hat, u, v, lam) < 0 else 0
    return -numerator/sigma_hat[u, u]/2 if lam*lam < numerator*numerator/(4*sigma_hat[u,u]) else 0


def gamma_hat_diag(Gamma, sigma_hat, u):
    # star = sum([Gamma[j, u]*(sigma_hat[j, u]+sigma_hat[u, j]) for j in range(len(Gamma)) if j != u])
    star = sum([2 * Gamma[j, u] * sigma_hat[j, u]for j in range(len(Gamma)) if j != u])
    return (-star + np.sqrt(star*star + 16*sigma_hat[u, u]))/(4*sigma_hat[u, u])

def gamma_hat_diag_same(Gamma, sigma_hat):
    m = Gamma.shape[0]
    Gamma_off = Gamma * ~np.eye(m, dtype=bool)
    tr_Gamma_off_sigma = np.trace(Gamma_off@sigma_hat)
    tr_sigma = np.trace(sigma_hat)
    d = (-tr_Gamma_off_sigma + np.sqrt(tr_Gamma_off_sigma**2 + 4*m*tr_sigma))/(2*tr_sigma)
    return d


def CD(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2, start=None):
    """

    X: N by P data matrix
    """
    N, P = X.shape
    sigma = np.cov(X.T)
    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        Gamma[u, v] = 0
                    else:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj


def CD_order(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2, start=None):
    """

    X: N by P data matrix
    """
    TO = np.array(EqVarDAG_TD_internal(X)['TO'])
    original_order = np.argsort(TO)

    # Reorder the adjacency matrix
    moral = moral[TO, :][:, TO]
    X = X[list(X.columns[TO])]


    N, P = X.shape
    sigma = np.cov(X.T)
    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        Gamma[u, v] = 0
                    else:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)

    # back to the original order
    opt_Gamma = opt_Gamma[original_order, :][:, original_order]

    return opt_Gamma, min_obj


def CD_3(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2, start=None):
    """

    X: N by P data matrix
    """
    N, P = X.shape
    sigma = np.cov(X.T)
    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    if Gamma[u, v] == 0:
                        temp_gamma = Gamma.copy()
                        temp_gamma -= np.diag(np.diag(temp_gamma))
                        cycle_uv = cycle(temp_gamma, u, v)
                        if not cycle_uv:
                            Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                    else:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                        obj_add = objective(Gamma, sigma, lam)
                        temp_gamma = Gamma.copy()
                        temp_gamma[u, v] = 0
                        cycle_vu = cycle(temp_gamma, v, u)
                        if not cycle_vu:
                            temp_gamma[v, u] = gamma_hat(temp_gamma, sigma, v, u, lam)
                        obj_reverse = objective(temp_gamma, sigma, lam)
                        if obj_reverse < obj_add:
                            Gamma = temp_gamma.copy()
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj



def CD_block(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2, start=None):
    """

    X: N by P data matrix
    """
    N, P = X.shape
    sigma = np.cov(X.T)
    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            for v in range(u+1, P):
                if moral[u, v] == 1:
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    cycle_vu = cycle(temp_gamma, v, u)
                    if not cycle_uv and not cycle_vu:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                        Gamma_uv  = Gamma.copy()
                        obj_uv = objective(Gamma, sigma, lam)
                        Gamma[u, v] = 0
                        Gamma[v, u] = gamma_hat(Gamma, sigma, v, u, lam)
                        obj_vu = objective(Gamma, sigma, lam)
                        if obj_uv < obj_vu:
                            Gamma = Gamma_uv
                    elif cycle_uv and cycle_vu:
                        Gamma[u, v] = 0
                        Gamma[v, u] = 0
                    elif cycle_uv:
                        Gamma[u, v] = 0
                        Gamma[v, u] = gamma_hat(Gamma, sigma, v, u, lam)
                    else:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                        Gamma[v, u] = 0
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj


def CD_consistent(X, moral, lam=0.01, cov=None, MAX_cycles=100, tol=1e-2, start=None):
    """

    X: N by P data matrix
    """
    N, P = X.shape
    sigma = np.cov(X.T) if cov is None else cov
    # sigma = (X.T@X/N).values
    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = float('inf')
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        if t % 10 == 0:
            print(f"cycle: {t}")
        for u in range(P):
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        Gamma[u, v] = 0
                    else:
                        Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
        np.fill_diagonal(Gamma, gamma_hat_diag_same(Gamma, sigma))
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    np.fill_diagonal(Gamma, gamma_hat_diag_same(Gamma, sigma))
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t < tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)
        # print(opt_Gamma)

    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj

def best_CD_consistent(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2, start=None):
    N, P = X.shape
    sigma = np.cov(X.T)

    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = objective(Gamma, sigma, lam)
    opt_Gamma = None
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        # if t % 10 == 0:
        #     print(f"cycle: {t}")
        improve = 0
        nxt_Gamma = Gamma.copy()
        candidate_Gamma = Gamma.copy()
        for u in range(P):
            np.fill_diagonal(candidate_Gamma, gamma_hat_diag_same(candidate_Gamma, sigma))
            improve_tmp = min_obj - objective(candidate_Gamma, sigma, lam)
            if improve_tmp > improve:
                improve = improve_tmp
                nxt_Gamma = candidate_Gamma.copy()
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    candidate_Gamma = Gamma.copy()
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        candidate_Gamma[u, v] = 0
                    else:
                        candidate_Gamma[u, v] = gamma_hat(candidate_Gamma, sigma, u, v, lam)
                    improve_tmp = min_obj - objective(candidate_Gamma, sigma, lam)
                    if improve_tmp > improve:
                        improve = improve_tmp
                        nxt_Gamma = candidate_Gamma.copy()
        Gamma = nxt_Gamma.copy()
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 5:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    # Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
                    np.fill_diagonal(Gamma, gamma_hat_diag_same(Gamma, sigma))
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)


        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t <= tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)

    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj


def best_CD(X, moral, lam=0.01, MAX_cycles=100, tol=1e-2, start=None):
    N, P = X.shape
    sigma = np.cov(X.T)

    inv_sigma = np.linalg.pinv(sigma)

    Gamma = np.eye(P) if start is None else start
    objs = []
    min_obj = objective(Gamma, sigma, lam)
    opt_Gamma = Gamma.copy()
    support_counter = defaultdict(int)
    for t in range(MAX_cycles):
        # if t % 10 == 0:
        #     print(f"cycle: {t}")
        improve = 0
        nxt_Gamma = Gamma.copy()
        candidate_Gamma = Gamma.copy()
        for u in range(P):
            candidate_Gamma[u, u] = gamma_hat_diag(candidate_Gamma, sigma, u)
            improve_tmp = min_obj - objective(candidate_Gamma, sigma, lam)
            if improve_tmp > improve:
                improve = improve_tmp
                nxt_Gamma = candidate_Gamma.copy()
            for v in range(P):
                if u != v and moral[u, v] == 1:
                    candidate_Gamma = Gamma.copy()
                    temp_gamma = Gamma.copy()
                    temp_gamma -= np.diag(np.diag(temp_gamma))
                    cycle_uv = cycle(temp_gamma, u, v)
                    if cycle_uv:
                        candidate_Gamma[u, v] = 0
                    else:
                        candidate_Gamma[u, v] = gamma_hat(candidate_Gamma, sigma, u, v, lam)
                    improve_tmp = min_obj - objective(candidate_Gamma, sigma, lam)
                    if improve_tmp > improve:
                        improve = improve_tmp
                        nxt_Gamma = candidate_Gamma.copy()
        Gamma = nxt_Gamma.copy()
        obj_t = objective(Gamma, sigma, lam)

        support_i = str(np.array(Gamma != 0, dtype=int).flatten())
        support_counter[support_i] += 1

        # spacer step
        if support_counter[support_i] == 2:
            # print("spacer step is working")
            for u, v in np.transpose(np.nonzero(Gamma)):
                if u != v:
                    Gamma[u, v] = gamma_hat(Gamma, sigma, u, v, lam)
                else:
                    Gamma[u, u] = gamma_hat_diag(Gamma, sigma, u)
            support_counter[support_i] = 0
            obj_t = objective(Gamma, sigma, lam)




        # if len(objs) > 1 and obj_t == objs[-1]:
        if len(objs) > 1 and objs[-1] - obj_t <= tol:
            objs.append(obj_t)
            print(f"stop at the {t}-th iteration.")
            break
        if obj_t < min_obj:
            min_obj = obj_t
            opt_Gamma = Gamma.copy()
        objs.append(obj_t)

        ## change direction ##
    for u, v in np.transpose(np.nonzero(opt_Gamma)):
        if u != v:
            new_Gamma = Gamma.copy()
            new_Gamma[u, v] = 0
            if not cycle(new_Gamma, v, u):
                new_Gamma[v, u] = gamma_hat(new_Gamma, sigma, v, u, lam)
                if objective(new_Gamma, sigma, lam) < objective(opt_Gamma, sigma, lam):
                    opt_Gamma = new_Gamma
                    print("Change direction!!!!!!!!!")


    # print(objs)
    # print(objsi)
    return opt_Gamma, min_obj

def skeleton(dag):
    """
    Given a list of arcs in the dag, return the undirected skeleton.
    This is for the computation of SHDs
    :param dag: list or arcs with 0 or 1 entries
    :return: skeleton np.array
    """
    skeleton_array = np.array(dag) + np.array(dag).T
    return skeleton_array


def compute_SHD(learned_DAG, True_DAG, SHDs=False):
    """
    Compute the stuctural Hamming distrance, which counts the number of arc differences (
    additions, deletions, or reversals)

    :param learned_DAG: list of arcs, represented as adjacency matrix
    :param True_DAG: list of arcs
    :return: shd: integer, non-negative
    """
    if type(learned_DAG) == tuple:
        learned_DAG = learned_DAG[0]
    if type(True_DAG) == tuple:
        True_DAG = True_DAG[0]
    learned_arcs = mat2ind(learned_DAG, len(learned_DAG))
    true_arcs = mat2ind(True_DAG, len(True_DAG))
    learned_skeleton = learned_arcs.copy()
    for item in learned_arcs:
        learned_skeleton.append((item[1], item[0]))
    True_skeleton = true_arcs.copy()
    for item in true_arcs:
        True_skeleton.append((item[1], item[0]))

    shd1 = len(set(learned_skeleton).difference(True_skeleton)) / 2
    shd2 = len(set((True_skeleton)).difference(learned_skeleton)) / 2
    Reversed = [(y, x) for x, y in learned_arcs]
    shd3 = len(set(true_arcs).intersection(Reversed))

    shd = shd1 + shd2 + shd3
    if SHDs:
        return shd1 + shd2
    return shd


def mat2ind(mat, p):
    edges = [(i, j) for i in range(p) for j in range(p) if mat[i][j] == 1]
    return edges


if __name__ == '__main__':
    results_1 = []
    results_2 = []
    results_3 = []
    runs = 10
    # do 1000 tests
    for _ in range(runs):
        # generate 50000 samples on each iteration
        # from X -> Z <- Y
        n=1000000
        X = np.random.normal(0, 1, n)
        Y = np.random.normal(0, 1, n)
        Z = X + Y + np.random.normal(0, 1, n)
        # Z = X + Y*5

        # in one data set have the data respect causal ordering
        # in the other have Z appear before X and Y
        # everything else about the data is the same
        data_1 = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        data_2 = pd.DataFrame({"Z": Z, "Y": Y, "X": X})
        data_3 = pd.DataFrame({"X": X, "Z": Z, "Y": Y})
        N, P = data_1.shape

        # true moral graph is fully connected
        true_moral = np.ones((P, P)) - np.eye(P)

        # get estimates of the DAG from both datasets
        # est_1, min_obj1 = CD(data_1, true_moral, MAX_cycles=400, lam=np.sqrt(12*np.log(P)/N))

        est_1, min_obj1 = CD_block(data_1, true_moral, MAX_cycles=400, lam=np.sqrt(12*np.log(P)/N)*10)
        est_1_ = np.array([[1 if i != j and est_1[i, j] != 0 else 0 for j in range(P)] for i in range(P)])

        # est_2, min_obj2 = best_CD(data_2, true_moral, MAX_cycles=400, lam=np.sqrt(12 * np.log(P)/N)/10)
        # st_2, min_obj2 = best_CD(data_2, true_moral, MAX_cycles=400, lam=np.sqrt(12 * np.log(P)/N))
        est_2, min_obj2 = CD_block(data_2, true_moral, MAX_cycles=400, lam=np.sqrt(12*np.log(P)/N)*10, tol=0)
        est_2_ = np.array([[1 if i != j and est_2[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
        # est_3, min_obj3 = CD(data_3, true_moral, MAX_cycles=400, lam=np.sqrt(12*np.log(P)/N)*10, tol=0)
        # est_3_ = np.array([[1 if i != j and est_3[i, j] != 0 else 0 for j in range(P)] for i in range(P)])

        # count the number of edges in each dataset
        # if collider bias is a problem for coordinate descent if the
        # data is not providing according to a causal ordering, then
        # est_2 will have 3 edges while est_1 will have 2 edges
        results_1.append(np.sum(est_1_) == 2)
        results_2.append(np.sum(est_2_) == 2)
        # results_3.append(np.sum(est_3_) == 2)
        # print(est_1_, est_2_, est_3_)

    print(np.sum(results_1)/runs)
    print(np.sum(results_2)/runs)
    # print(np.sum(results_3)/runs)
    print(est_1_)
    print(est_2_)
    # print(est_3_)


    # data, true_dag, moral_lasso, true_moral = read_data("6cloud", n=10000, iter=1)
    # N, P = data.shape
    # true_moral = true_moral + true_moral.T
    #
    # reverse_order = np.arange(len(true_dag) - 1, -1, -1)
    # # Reorder the adjacency matrix
    # new_true_dag = true_dag[reverse_order, :][:, reverse_order]
    # new_true_moral = true_moral[reverse_order, :][:, reverse_order]
    # new_data = data[list(data.columns[::-1])]
    #
    # # ones = np.ones((P, P))
    # # ones = ones - np.diag(np.diag(ones))
    # start_i = time.time()
    # est, min_obj = best_CD(new_data, new_true_moral, MAX_cycles=400, lam=np.sqrt(12*np.log(P)/N), tol=0) # np.sqrt(12*np.log(P)/N) np.sqrt(np.log(N)/np.sqrt(N))/2
    # end_i = time.time()
    # est_ = np.array([[1 if i != j and est[i, j] != 0 else 0 for j in range(P)] for i in range(P)])
    #
    # true_dag_ = cd.DAG.from_amat(new_true_dag)
    # true_cpdag = true_dag_.cpdag().to_amat()
    # estimated_dag = cd.DAG.from_amat(est_)
    # estimated_cpdag = estimated_dag.cpdag().to_amat()
    # SHD_cpdag = np.sum(np.abs(estimated_cpdag[0] - true_cpdag[0]))
    #
    # skeleton_estimated, skeleton_true = skeleton(est_), skeleton(new_true_dag)
    # SHDs = compute_SHD(skeleton_estimated, skeleton_true, True)
    # TPR = np.sum(np.logical_and(est_, new_true_dag))/np.sum(new_true_dag)
    # FPR = (np.sum(est_) - np.sum(np.logical_and(est_, new_true_dag)))/(P*P - np.sum(new_true_dag))
    # print(f"TPR: {TPR}; FPR:{FPR}; shd_cpdag: {SHD_cpdag}; SHDs: {SHDs}; obj:{min_obj}; Time:{np.round(end_i - start_i, 3)}")