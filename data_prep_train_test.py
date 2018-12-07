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
data_train.iloc[:, 5:20].head()

main_test = main_test.rename(index=str, columns={'event': 'event_id'})
data_test = pd.merge(main_test, events, on='event_id')

data_train.to_csv('.\\data\\data_train.csv')
data_test.to_csv('.\\data\\data_test.csv')
main_train.to_csv('.\\data\\data_main_train.csv')
main_test.to_csv('.\\data\\data_main_test.csv')

import pandas as pd
data_train = pd.read_csv('.\\data\\data_train.csv')
data_train.drop(data_train.columns[0],axis=1)