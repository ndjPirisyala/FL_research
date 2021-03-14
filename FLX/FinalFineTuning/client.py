import pandas as pd
import tensorflow as tf
import sys

name = sys.argv[1]
client = int(sys.argv[2].strip())

data = pd.read_csv('tra.csv')
# data = data.loc[data.digit==client].sample(n=100, random_state=1)
# data = data.loc[data.digit==client].iloc[:100,:]
# data = data.sample(n=1000, random_state=1)
data = data.iloc[:1000,:]
data.drop(data.filter(regex="Unname"),axis=1, inplace=True)

load = tf.keras.models.load_model('TEST')
load.fit(data.iloc[:,:-2], data.digit, epochs=25, batch_size = 32)

load.save('Local-{}-{}'.format(name,client))