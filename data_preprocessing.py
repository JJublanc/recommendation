from sklearn.model_selection import train_test_split
import pandas as pd

# configuration du chemin d'accès
import os

path = '.\\data'
os.listdir(path)

# chargement des tables du projet
main = pd.read_csv('.\\data\\train.csv')
test = pd.read_csv(".\\data\\test.csv")
users = pd.read_csv(".\\data\\users.csv")
user_friends = pd.read_csv(".\\data\\user_friends.csv")
events = pd.read_csv(".\\data\\events.csv")
event_attendees = pd.read_csv(".\\data\\event_attendees.csv")

# on split les données de la table principale
main_train, main_test = train_test_split(main, test_size=0.2, random_state=28)

# description de la table
events_by_user = main.assign(n=0).groupby('user').n.count().head().reset_index(name='nb_event')
events_by_user.nb_event.plot.box()
events_by_user.assign(n=0).groupby('nb_event').n.count().plot.line()

# inclusion des event et user dans les tables tests et trains
len(test.user.unique())
user_in_train_test = [ii for ii in range(len(test.user.unique())) if test.user.unique()[ii] in main.user.unique()]
len(user_in_train_test)

event_in_train_test = [ii for ii in range(len(test.event.unique())) if test.event.unique()[ii] in main.event.unique()]
test.event.unique()
len(event_in_train_test)

# frequence des événements  en fonction du nombre de user interessés
main_nb_interested = main_train.groupby('event_id').interested.sum().reset_index(name='nb_interested')
main_nb_interested.assign(n=0).groupby('nb_interested').n.count().reset_index(name='frequence').frequence.plot.line()


