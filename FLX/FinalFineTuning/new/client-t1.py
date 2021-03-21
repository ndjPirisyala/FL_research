import pandas as pd
import tensorflow as tf
import sys
import random
import pandas as pd
from functools import reduce
from sklearn.utils import shuffle
import pickle

name = sys.argv[1]
client = int(sys.argv[2].strip())

data = pd.read_csv('tra.csv')

# data = data.loc[data.digit==client].sample(n=100, random_state=1)
# data = data.sample(n=1000, random_state=1)
# data = data.loc[data.digit==client].iloc[:100,:]
# data = data.iloc[:1000,:]

# l=random.sample(list(range(10)),random.choice(list(range(1,10))))
l=list(range(int(client)+1))
data = reduce(lambda df1, df2: df1.merge(df2, "outer"), [data.loc[data.digit==i] for i in l])
data = shuffle(data)
data.reset_index(inplace=True, drop=True)
n=random.choice(list(range(100,201)))
data = data.sample(n=n, random_state=1)
data.drop(data.filter(regex="Unname"),axis=1, inplace=True)
# data.to_csv('{}.csv'.format(client))
# div=0
# for i in l:
#     div=div+abs((n/10)-len(data.loc[data.digit==i]))
# pickle.dump(div,open('{}.pkl'.format(client),'wb'))

# load = tf.keras.models.load_model('TEST')
import tensorflow as tf

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10,activation="sigmoid")
    ])

model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(data.iloc[:,:-2], data.digit, epochs=25, batch_size = 32)

model.save('Local-{}-{}'.format(name,client))