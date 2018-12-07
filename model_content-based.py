import pandas as pd
import numpy as np

# TODO ajouter les friends pour augmenter le data set
# TODO ajouter d'autres variables pour enrichir les modèles

# on importe les données

data_train = pd.read_csv('.\\data\\data_train.csv')
data_test = pd.read_csv('.\\data\\data_test.csv')

data_train = data_train.drop(data_train.columns[0], axis=1)
data_test = data_test.drop(data_test.columns[0], axis=1)

# ______________________
#   partie predictions
# ______________________

# import packages
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# calcul des predictions
columns = [data_train.columns[0]] + list(data_train.columns[range(14, 115)])

predictionsSVC = []
predictionsNN = []
predictionsDTree = []
predictionsNBayes = []

ii = 0

for ii in data_test.index:
    user = data_test.loc[ii, 'user']
    event = data_test.loc[ii, 'event_id']
    X = data_train.loc[data_train.user == user, columns]
    y = data_train[data_train.user == user]['interested']
    predictionSVC = []
    predictionNN = []
    predictionDTree = []
    predictionNBayes = []

    if len(y.unique()) == 1:
        predictionSVC.append(y.unique()[0])
        predictionNN.append(y.unique()[0])
        predictionDTree.append(y.unique()[0])
        predictionNBayes.append(y.unique()[0])
    elif len(y.unique()) == 0:
        predictionSVC.append(np.nan)
        predictionNN.append(np.nan)
        predictionDTree.append(np.nan)
        predictionNBayes.append(np.nan)
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
        classifDT = DecisionTreeClassifier(random_state=0)
        classifDT.fit(X, y)
        predictionDTree = classifDT.predict(data_test.loc[data_test.event_id == event, columns])

        # partie Naive Bayes
        classifNB = GaussianNB()
        classifNB.fit(X, y)
        predictionNBayes = classifNB.predict(data_test.loc[data_test.event_id == event, columns])

    # add to predictions
    predictionsSVC.append(predictionSVC[0])
    predictionsNN.append(predictionNN[0])
    predictionsDTree.append(predictionDTree[0])
    predictionsNBayes.append(predictionNBayes[0])

data_test['pred_SVC'] = predictionsSVC
data_test['pred_NN'] = predictionsNN
data_test['pred_DT'] = predictionsDTree
data_test['pred_NBayes'] = predictionsNBayes

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
