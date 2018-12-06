import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

# import des data
events = pd.read_csv(".\\data\\events.csv")
events.head()

# on remplace les NaN par des vides
events = events.fillna('')

# on ne garde que les colonnes desciptives des événements
col = [0]
for ii in range(9, 110):
    col.append(ii)
events_content = events.iloc[:, col]
events_content.shape

# on calcule les distances cosinus entre les événements (ici seulement les 1000 premiers)
events_cosine_similarity_1000_first = linear_kernel(events_content.iloc[1:1000, ], events_content.iloc[1:1000, ])
events_cosine_similarity_1000_first.shape
events_cosine_similarity_1000_first[0]

# On crée un pd.Series pour indexer les event_id
indices = pd.Series(events_content.index, index=events_content['event_id'].drop_duplicates())

# fonction renvoyant les 10 items les plus proches

def get_recommendations(event_id, cosine_sim = events_cosine_similarity_1000_first):
    # on récupère l'indice de l'événement
    idx = indices[event_id]

    # on récupère les pairs (index,score) des similarité de l'event_id avec les autres events
    sim_scores = list(enumerate(cosine_sim[idx]))

    # on trie de manière décroissante les similarités
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # on récupère le score pour les 10 événements les plus similaire (le premier est l'événement lui-même, il est donc
    # éliminé
    sim_scores = sim_scores[1:11]

    # on retourne l'indice des events les plus proches
    events_indices = [i[0] for i in sim_scores]

    # renvoie les identifiants des 10 plus proches événements
    return events_content['event_id'].iloc[events_indices]

get_recommendations(events_content.loc[500,'event_id'])





