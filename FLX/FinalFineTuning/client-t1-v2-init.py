import pandas as pd
import tensorflow as tf
import sys

name = sys.argv[1]
client = int(sys.argv[2].strip())

data = pd.read_csv('tra.csv')
data = data.loc[data.digit==client].sample(n=100, random_state=1)
# data = data.loc[data.digit==client].iloc[:100,:]
data.drop(data.filter(regex="Unname"),axis=1, inplace=True)

# load = tf.keras.models.load_model('TEST')
import tensorflow as tf

# model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(16, activation='relu'),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(10,activation="sigmoid")
#     ])
model = tf.keras.models.load_model('TESTV2')

model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(data.iloc[:,:-2], data.digit, epochs=25, batch_size = 32)

model.save('Local-{}-{}'.format(name,client))