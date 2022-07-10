import sys
import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import sklearn
import heapq
import time
import decimal
import operator
import functools
from datetime import timedelta
from sklearn.model_selection \
    import train_test_split
from sklearn.linear_model \
    import LogisticRegression
from sklearn.metrics import confusion_matrix, \
    accuracy_score, recall_score, \
    precision_score, classification_report
from sklearn import model_selection, preprocessing


def read(filename):
    # Stocker les données dans une variable
    data = pd.read_csv(filename, delimiter=";", index_col=None)
    # Affichage des données
    print(data)
    # Renvoie les données à la fin de la fonction
    return data

def count_features(data, target):
    copy = data.copy()
    copy = copy.drop([target], axis=1)

    n=len(copy.columns)

    return n

def learning(data, target):
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Initialise une matrice carrée de zéros de taille 2
    matrix = np.zeros((2, 2), dtype=float)

    model = LogisticRegression(solver='liblinear')

    matrix, y_test, y_pred = cross_validation(5, X, y, model, matrix)

    return accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

def cross_validation(nfold, X, y, model, matrix):
    k = model_selection.KFold(nfold)

    y_test_lst = []
    y_pred_lst = []

    # Permet de séparer les données en k répartitions pour chaque répartition on effectue un apprentissage
    for train_index, test_index in k.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Somme des matrices de confusions pour chacune des répartitions
        matrix = matrix + confusion_matrix(y_test, y_pred)

        # Ajout des valeurs réelles (y_test) dans la liste
        y_test_lst.extend((y_test))

        # Ajout des valeurs prédites par le modèle (y_pred) dans une autre liste
        y_pred_lst.extend((y_pred))

    return matrix, y_test_lst, y_pred_lst


def preparation(data, features, target):

    copy = data.copy()
    copy_target = copy[target]
    copy = copy.drop([target], axis=1)

    le = preprocessing.LabelEncoder()

    for column_name in copy.columns:
        if copy[column_name].dtype == object:
            copy[column_name] = le.fit_transform(copy[column_name])
    else:
        pass

    # Récupère les données correspondantes
    copy = copy.iloc[:, features]
    copy = pd.DataFrame(copy)
    copy[target] = copy_target

    return copy

def FitnessFunction(data, target, alpha, n, solution):
    sol_array = np.array(solution)
    for elements in sol_array:
        nb_zeroes = np.count_nonzero(elements == 0)

    unselected_features = nb_zeroes / n

    non_zero_indices = np.nonzero(sol_array)
    features = []

    for i in range(0, len(non_zero_indices[0])):
        features.append(non_zero_indices[0][i])

    cols_selection = preparation(data, features, target)
    accuracy = learning(cols_selection,target)

    return (alpha*accuracy-(1-alpha)*unselected_features)

def tri(lst):
    lst_tri = sorted(lst)
    index_lst_tri = sorted(range(len(lst)), key=lambda k: lst[k])

    return lst_tri, index_lst_tri

def equation1(X, lb, ub, n, d):
    for i in range(n):
        for j in range(d):
            X[i][j] = lb + np.random.rand() * (ub - lb)

    return X

def equation1_v2(Xij, lb, ub):

    Xij = lb + np.random.rand() * (ub - lb)

    return Xij

def equation8(X, n, d):
    S = np.zeros(shape=(n, d), dtype=float)
    X_new = np.zeros(shape=(n, d), dtype=float)
    for i in range(n):
        for j in range(d):
            S[i][j] = 1/(1+np.exp(-X[i][j]))
            if S[i][j] < np.random.rand():
                X_new[i][j] = 1
            else:
                X_new[i][j] = 0

    return X_new


def BBACE(data, target, n, d, Max_iteration, FitnessFunction):

    alpha = 0.9
    gamma = 0.9
    lb = 0
    ub = 1
    thresh = 0.5
    fmin = 0
    fmax = 2
    A_max = 2
    r0_max = 1
    A = np.random.uniform(1, A_max, n)
    r0 = np.random.uniform(0, r0_max, n)
    r = r0.copy()
    p = 0.2
    t = 0
    t_list = []
    t_list.append(t)

    X = np.zeros(shape=(n,d), dtype=float)
    v = np.zeros(shape=(n, d), dtype=float)
    fitness = np.zeros(shape=(n,1), dtype=float)

    for i in range(0,n):
        for j in range(0,d):
            if np.random.rand()>thresh:
                X[i][j] = 1
            else:
                X[i][j] = 0

    X_assessed = equation1(X, lb, ub, n, d)

    for i in range(0,n):
        fitness[i,0] = FitnessFunction(data, target, alpha, n, X_assessed[i,:])
    fitness = fitness[:,0].tolist()
    fitness_tri, idx_fitness_tri = tri(fitness)
    best_fitness = fitness_tri[-1]
    best_X_assessed_idx = idx_fitness_tri[-1]
    best_X_assessed = X_assessed[best_X_assessed_idx, :]

    score = np.zeros([1, Max_iteration], dtype='float')
    score[0,t] = best_fitness
    print("Generation:", t + 1)
    print("Best (BA):", score[0,t])

    t += 1

    while t < Max_iteration:

        Ne = math.ceil(p * n)

        X_new = np.zeros(shape=(n,d), dtype=float)

        for i in range(0, n):
            f = fmin + (fmax - fmin) * np.random.rand()
            for j in range(0, d):
                v[i,j] = v[i,j] + (best_X_assessed[j] - X_assessed[i,j]) * f
                X_new[i,j] = X_assessed[i,j] + v[i,j]
                X_new[i,j] = equation1_v2(X_new[i][j], lb, ub)

            if np.random.rand() > r[i]:
                for j in range(0,d):
                    eps = -1 + 2 * np.random.rand()
                    X_new[i,j] = best_X_assessed[j] + eps * np.mean(A)
                    X_new[i,j] = equation1_v2(X_new[i][j], lb, ub)
            else:
                pass

        X_assessed_new = equation8(X_new, n, d)
        for i in range(0,n):
            sum = 0
            for j in range(0,d):
                if X_assessed_new[i][j] == 0:
                    sum = sum + 1
            if sum == d:
                rand = random.randint(0, d)
                X_assessed_new[i,rand] = 1

        for i in range(0,n):
            Fnew = FitnessFunction(data, target, alpha, n, X_assessed_new[i, :])
            if (np.random.rand() < A[i]) and (Fnew > fitness[i]):
                X_assessed[i, :] = X_assessed_new[i, :]
                fitness[i] = Fnew
                # Loudness update (6)
                A[i] = alpha * A[i]
                # Pulse rate update (6)
                r[i] = r0[i] * (1 - np.exp(-gamma * t))
            else:
                pass

        fitness_tri, idx_fitness_tri = tri(fitness)
        best_fitness = fitness_tri[-1]
        best_X_assessed_idx = idx_fitness_tri[-1]
        best_X_assessed = X_assessed[best_X_assessed_idx, :]

        X_assessed = pd.DataFrame(X_assessed)
        best_Ne_Bats_idx = idx_fitness_tri[-Ne:]
        best_Ne_bats = X_assessed.iloc[best_Ne_Bats_idx, :]
        X_assessed = np.array(X_assessed)
        best_Ne_bats = np.array(best_Ne_bats)

        Y = np.zeros([Ne, d], dtype='float')
        for i in range(0,Ne):
            for j in range(0,d):
                if np.random.rand() < best_Ne_bats[i][j]:
                    Y[i, j] = 1
                else:
                    Y[i, j] = 0

        Y_assessed = equation1(Y, lb, ub, Ne, d)

        worst_Ne_Bats_idx = idx_fitness_tri[:Ne]
        X_assessed = pd.DataFrame(X_assessed)
        worst_Ne_bats = X_assessed.iloc[worst_Ne_Bats_idx, :]
        X_assessed = np.array(X_assessed)
        worst_Ne_bats = np.array(worst_Ne_bats)

        for i in range(0,len(worst_Ne_Bats_idx)):
            X_assessed[worst_Ne_Bats_idx[i], :] = Y_assessed[i, :]

        for i in range(0, n):
            Fnew = FitnessFunction(data, target, alpha, n, X_assessed[i, :])
            if (Fnew > fitness[i]):
                fitness[i] = Fnew

        fitness_tri, idx_fitness_tri = tri(fitness)
        best_fitness = fitness_tri[-1]
        best_X_assessed_idx = idx_fitness_tri[-1]
        best_X_assessed = X_assessed[best_X_assessed_idx, :]

        score[0,t] = best_fitness
        print("Generation:", t + 1)
        print("Best (BA):", score[0,t])
        t += 1
        t_list.append(t)

    Gbin = best_X_assessed
    Gbin = Gbin.reshape(d)
    pos = np.asarray(range(0, d))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    preparation1 = preparation(data, sel_index, target)
    accuracy = learning(preparation1, target)

    plt.figure("Evolution du score selon le temps")
    plt.plot(t_list, score[0,:].tolist())
    plt.title(' Evolution du score selon le temps')
    plt.xlabel(' Temps ')
    plt.ylabel(' Score ')
    plt.show()

    # Create dictionary
    ba_data = {'features_selection': sel_index.tolist(), 'accuracy': accuracy, 'number_features': num_feat}

    return ba_data


if __name__ == '__main__':
    d = read("framingham3.csv")
    target_name = "TenYearCHD"

    n = count_features(d, target_name)

    Max_iteration = 50  # max no. of iterations
    noP = 30  # No. of artificial bats
    noV = n # dimension of search variables

    BBACE_FS = BBACE(d, target_name, noP, noV, Max_iteration, FitnessFunction)
    print(BBACE_FS)
