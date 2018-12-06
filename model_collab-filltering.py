from data_preprocessing import *
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

# __________________
# preparation data
# __________________

# on créer des identifiants int pour les users et events
main_train_save = main_train
main_test_save = main_test

# pour réinitialiser les données sans avoir à rappeler data_preprocessing
# main_train = main_train_save
# main_test = main_test_save


main_train = main_train.assign(user_id=main_train.user.astype("category").cat.codes)
main_train = main_train.assign(event_id=main_train['event'].astype("category").cat.codes)

# on créer deux tables de passage pour récupérer ensuite les identifiants et noms originaux des users et events
user_lookup = main_train[['user_id', 'user']].drop_duplicates()
user_lookup['user_id'] = user_lookup.user_id.astype(str)

event_lookup = main_train[['event_id', 'event']].drop_duplicates()
event_lookup['event_id'] = event_lookup.event_id.astype(str)

# on ne garde que les données concernant les user_id event_id et le fait que l'utilisateur est intéressé ou non
main_train = main_train.drop(['user', 'event', 'invited', 'timestamp', 'not_interested'], axis=1)

# On regroupe les valeurs de interested dans interested_score (si un user est intéressé plusieur fois, on somme
# le nombre de fois où il s'est montré intéressé
main_train = (main_train.groupby(['user_id', 'event_id'])
              .interested
              .sum()
              .sort_values(ascending=False)
              .reset_index(name='interested_score'))

# verification : on constate qu'il n'y a aucun event pour lequel un user s'est dit intéressé plusieurs fois
main_train.groupby('interested_score').size()

# on élimine les valeurs nulles
main_train = main_train[main_train.interested_score.notnull()]

# on récupère ensuite les lignes et colonnes pour la nouvelle matrice

rows = main_train.user_id.astype(int)
cols = main_train.event_id.astype(int)
scores = np.array(main_train.interested_score)

data_sparse = sparse.csr_matrix((scores, (rows, cols)))

# _____________
##  modele SVD
# _____________

from scipy.sparse.linalg import svds

# calcul des matrices
u, s, vt = svds(data_sparse.astype(float), k=10)
svd_matrix = u.dot(np.diag(s)).dot(vt)


# fonction de prediction
def prediction_SVD(user_test, event_test):
    user_test_id = user_lookup[user_lookup.user == user_test].user_id.astype(int)
    event_test_id = event_lookup[event_lookup.event == event_test].event_id.astype(int)
    try:
        result = float(svd_matrix[user_test_id, event_test_id])
    except TypeError:
        return float('NaN')
    return result


# on applique la fonction de prediction aux données testées
main_test['interested_pred_SVD'] = main_test.apply(lambda row: prediction_SVD(row.user, row.event), axis=1)

# on construit une courbe ROC pour mesurer la performance du modèle
from sklearn.metrics import roc_curve

fpr = []
tpr = []
tbl = main_test[main_test.loc[:, 'interested_pred_SVD'].notnull()]
fpr, tpr, _ = roc_curve(tbl.loc[:, 'interested'].values, tbl.loc[:, 'interested_pred_SVD'].values)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr)
