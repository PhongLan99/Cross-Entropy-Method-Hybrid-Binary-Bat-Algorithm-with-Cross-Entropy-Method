import numpy as np
import pandas as pd
import random
import math
import scipy.stats as st
import time
from datetime import timedelta
import matplotlib.pyplot as plt


def read(filename):
    # Stocker les données dans une variable
    data = pd.read_csv(filename, delimiter=",", index_col=None)
    # Affichage des données
    print(data)
    # Renvoie les données à la fin de la fonction
    return data


def vecteur_weight_profit(data):
    weight = data.iloc[:, 0].tolist()
    profit = data.iloc[:, 1].tolist()

    return weight, profit


def longueur(vecteur):
    n = len(vecteur)

    return n


def simulation_multivariate_bernoulli(v):
    U = np.random.rand()
    if U < v:
        X = 1
    else:
        X = 0

    return X


def weight_array(weight):
    wa = [weight] * len(weight)

    return wa


def indicatrice_1(weight, x, capacity):
    if sum(a * b for (a, b) in zip(weight, x)) > capacity:
        indicatrice = 1
    else:
        indicatrice = 0

    return indicatrice


def function_S(n, profit, weight, X):
    Si = []
    for i in range(n):
        Si.append(indicatrice_1(weight, X, capacity))

    S = (-sum(profit) * sum(Si)) + sum(a * b for (a, b) in zip(profit, X))

    return S


def performance_S(N, n, variables):
    S_list = []
    for i in range(N):
        S = function_S(n, profit, weight, variables[i])
        S_list.append(S)

    return (S_list)


def tri(S_list):
    S_list_tri = sorted(S_list)
    index_list_tri = sorted(range(len(S_list)), key=lambda k: S_list[k])

    return S_list_tri, index_list_tri


def indicatrice_2(N, S_list_tri, yt):
    s_indicatrices = []
    for k in range(N):
        if S_list_tri[k] >= yt:
            s_indicatrices.append(1)
        else:
            s_indicatrices.append(0)

    return s_indicatrices


def total_weight(w, n):
    S = 0
    for i in range(n):
        if w[1][i] == 1:
            S += w[0][i]
        else:
            S = S
    return S

def calcul_vt(vt, op2, s_indicatrices, variables):
    op1_list = []
    for m in range(n):
        op1 = []
        for l in range(N):
            op1.append(s_indicatrices[l] * variables[l][m])
        op1_list.append(sum(op1))
        vt.append(op1_list[m] / op2)
    return vt


def CE_backpack(capacity, N, n, p, weight, profit):
    debut = time.time()

    v = [1 / 2] * n
    Ne = math.ceil(p * N)
    w = weight
    t = 1
    y_tps = [t]
    dt = 0.1
    dt_list = []
    vt_list = []
    list_variables = []
    S_list_list = []
    S_list_tri_list = []
    index_list_tri_list = []
    variables_tri_list = []
    index_yt_list = []
    yt_list = []
    s_indicatrices_list = []
    S_total_list = []

    variables = []
    for i in range(N):
        X = []
        for j in range(n):
            X.append(simulation_multivariate_bernoulli(v[j]))
        variables.append(X)
    list_variables = variables

    S_list = performance_S(N, n, variables)
    S_list_list = S_list

    S_list_tri, index_list_tri = tri(S_list)
    S_list_tri_list, index_list_tri_list = S_list_tri, index_list_tri

    variables_tri = [i for _, i in sorted(zip(index_list_tri, variables))]
    variables_tri_list = variables_tri

    index_yt = N - Ne + 1
    index_yt_list = index_yt
    yt = S_list_tri[index_yt]
    yt_list = yt

    s_indicatrices = indicatrice_2(N, S_list, yt)
    s_indicatrices_list = s_indicatrices
    op2 = sum(s_indicatrices)

    vt = []
    vt=calcul_vt(vt, op2, s_indicatrices, variables)
    vt_list = vt

    c = []
    for k in range(n):
        c = np.array(1) - np.array(vt)

    for o in range(n):
        mini = min(vt[o], c[o])
        if mini == vt[o]:
            dt = max(vt)
        else:
            dt = max(c)
    dt_list = dt

    t = t + 1
    v = vt
    w = np.array(w)
    wa = np.vstack([w, v])

    S = total_weight(wa, n)
    S_total_list = S

    for i in range(1, 50):
        variables = []
        for i in range(N):
            X = []
            for j in range(n):
                X.append(simulation_multivariate_bernoulli(v[j]))
            variables.append(X)
        list_variables = np.vstack([list_variables, variables])

        S_list = performance_S(N, n, variables)
        S_list_list = np.vstack([S_list_list, S_list])

        S_list_tri, index_list_tri = tri(S_list)
        S_list_tri_list, index_list_tri_list = np.vstack([S_list_tri_list, S_list_tri]), np.vstack(
            [index_list_tri_list, index_list_tri])

        variables_tri = [i for _, i in sorted(zip(index_list_tri, variables))]
        variables_tri_list = np.vstack([variables_tri_list, variables_tri])

        index_yt = N - Ne + 1
        index_yt_list = np.vstack([index_yt_list, index_yt])
        yt = S_list_tri[index_yt]
        yt_list = np.vstack([yt_list, yt])

        s_indicatrices = indicatrice_2(N, S_list, yt)
        s_indicatrices_list = np.vstack([s_indicatrices_list, s_indicatrices])
        op2 = sum(s_indicatrices)

        vt = []
        vt=calcul_vt(vt, op2, s_indicatrices, variables)
        vt_list = np.vstack([vt_list, vt])

        c = []
        for k in range(n):
            c.append(np.array(1) - np.array(vt[k]))

        mini = []
        for o in range(n):
            mini.append(min(vt[o], c[o]))
            if mini[o] == vt[o]:
                dt = max(vt)
            else:
                dt = max(c)
        dt_list = np.vstack([dt_list, dt])

        t = t + 1
        y_tps.append(t)
        v = vt
        wa = np.vstack([w, v])

        S = total_weight(wa, n)
        S_total_list = np.vstack([S_total_list, S])

    t_max_yt_index = np.argmax(yt_list)
    t_max_yt = y_tps[t_max_yt_index]

    tps = timedelta(seconds=time.time() - debut)
    print(" temps: " + str(tps))

    plt.figure("Evolution du score selon le temps")
    plt.plot(y_tps, yt_list)
    plt.title(' Evolution du score selon le temps')
    plt.xlabel(' Temps ')
    plt.ylabel(' Score ')
    plt.show()

    return S_list, S_list_tri, yt, v, S, t_max_yt


if __name__ == '__main__':
    d = read("knapsack_test2.csv")

    weight, profit = vecteur_weight_profit(d)
    capacity = 9906309440
    n = longueur(weight)
    p = 0.1
    N = 20 ** 3
    S_list, S_list_tri, yt, v, S, t_max_yt = CE_backpack(capacity, N, n, p, weight, profit)
    print(" Score de la solution optimale: ", yt)
    print(" Vecteur de la solution optimale: ", v)
    print(" Poids total du sac: ", S)
    print(" Indice du temps du meilleur score : ", t_max_yt)







    
