import pandas as pd
from data_preprocessing import *
import random
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

# __________________
# preparation data
# __________________

# chargement de la table principale
main = pd.read_csv('.\\data\\train.csv')
events = pd.read_csv(".\\data\\events.csv")
# on liste les événements et les users
event_list = list(main.event.unique())
user_list = list(main.user.unique())

# on réalise le split sur les événements. l'idée est de réaliser des prédictions pour des événements nouveaux.
# on split les événement en deux, une partie pour le train une autre pour les tests

event_list_train = event_list[:round(len(event_list) * 0.8)]
event_list_test = event_list[round(len(event_list) * 0.8):len(event_list)]
# vérification que chaque événement est dans l'un des set
(len(event_list_test) + len(event_list_train)) == len(event_list)

# on divise les données en un jeu de train et un jeu de test
main_train = main[main.event.isin(event_list_train)]
main_test = main[main.event.isin(event_list_test)]

# on regroupe les données avec celles donnant des informations sur les événements
main_train = main_train.rename(index=str, columns={'event': 'event_id'})
data_train = pd.merge(main_train, events, on='event_id')

main_test = main_test.rename(index=str, columns={'event': 'event_id'})
data_test = pd.merge(main_test, events, on='event_id')

# on récupère les informations pour réaliser un premier test
idx_test = random.choice(main_test.index)
user_test = main_test.loc[main_test.index == idx_test, 'user'].values[0]
event_test = main_test.loc[main_test.index == idx_test, 'event_id'].values[0]

main_test[main_test.event_id == event_test]
main_train[main_train.user == user_test]

# TODO ajouter les friends pour augmenter le data set
# TODO ajouter d'autres variables pour enrichir les modèles

# _____________________
#  partie predictions
# _____________________

# import packages
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# calcul des predictions
columns = [data_train.columns[0]] + list(data_train.columns[range(14, 115)])
predictionsSVC = []
predictionsNN = []

for ii in data_test.index:
    user = data_test.loc[ii, 'user']
    event = data_test.loc[ii, 'event_id']
    X = data_train.loc[data_train.user == user, columns]
    y = data_train[data_train.user == user]['interested']
    predictionSVC = []
    predictionNN = []

    if len(y.unique()) == 1:
        predictionSVC.append(y.unique()[0])
        predictionNN.append(y.unique()[0])
    elif len(y.unique()) == 0:
        predictionSVC.append(np.nan)
        predictionNN.append(np.nan)
    else:
        # partie SVC
        classifSVC = SVC(gamma='auto')
        classifSVC.fit(X, y)
        predictionSVC = classifSVC.predict(data_test.loc[data_test.event_id == event, columns])

        # partie Nearest Neighboor
        neigh = KNeighborsClassifier(n_neighbors=2)
        neigh.fit(X, y)
        predictionNN = neigh.predict(data_test.loc[data_test.event_id == event, columns])

        # partie arbre de decision

    # add to predictions
    predictionsSVC.append(predictionSVC[0])
    predictionsNN.append(predictionNN[0])

data_test['pred_SVC'] = predictionsSVC
data_test['pred_NN'] = predictionsNN

# scores
score_columns = ['model_name', 'TP', 'FP', 'FN', 'precision', 'recall', 'F1']
scores = pd.DataFrame(columns=score_columns)
i1 = data_test.columns.get_loc('pred_SVC')
iN = len(data_test.columns)

for ii in range(i1, iN):
    model_name = data_test.columns[ii]

    TP = sum((data_test.interested == data_test.iloc[:, ii]) & (data_test.iloc[:, ii] == 1))
    FP = sum((data_test.interested != data_test.iloc[:, ii]) & (data_test.iloc[:, ii] == 1))
    FN = sum((data_test.interested != data_test.iloc[:, ii]) & (data_test.iloc[:, ii] == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1 = 2 * (recall * precision) / (recall + precision)

    score = pd.DataFrame([[model_name, TP, FP, FN, precision, recall, F1]], columns=score_columns)
    scores = pd.concat([scores, score])

scores
