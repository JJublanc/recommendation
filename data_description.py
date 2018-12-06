import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.preprocessing import OneHotEncoder

import os
path ='.\\data'
os.listdir(path)


# table train

train = pd.read_csv('.\\data\\train.csv')
train.describe()
train.head()


# table test
test = pd.read_csv(".\\data\\test.csv")
test.describe()
test.train()

# table users
users = pd.read_csv(".\\data\\users.csv")
users.head()

# table users_friends
user_friends = pd.read_csv(".\\data\\user_friends.csv")
user_friends.head()

# table events
events = pd.read_csv(".\\data\\events.csv")
events.head()

# table event_attendees
event_attendees = pd.read_csv(".\\data\\event_attendees.csv")
event_attendees.head()

