#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nest_asyncio
nest_asyncio.apply()


# In[ ]:


import collections

import copy 
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tff.backends.reference.set_reference_context()


# In[ ]:


from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models

import math
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from operator import add

import seaborn as sns
sns.set_style("whitegrid")


# In[ ]:


origTra = open('../Datasets/pendigits/pendigits-orig.tra', 'r').read()


# In[ ]:


listOrigTra = []
for i in origTra.split('COMMENT ')[1:]:
    listOrigTra.append(i.split()[1])
    
len(listOrigTra)


# In[ ]:


tra = open('../Datasets/pendigits/pendigits.tra', 'r').read()
listTra = []
for i in tra.split('\n'):
    listTra.append(i[-1])
    
len(listTra)


# In[ ]:


origTes = open('../Datasets/pendigits/pendigits-orig.tes', 'r').read()


# In[ ]:


listOrigTes = []
for i in origTes.split('COMMENT ')[1:]:
    listOrigTes.append(i.split()[1])
    
len(listOrigTes)


# In[ ]:


listOrigTes = [str(int(i)+30) for i in listOrigTes]


# In[ ]:


tes = open('../Datasets/pendigits/pendigits.tes', 'r').read()
listTes = []
for i in tes.split('\n'):
    listTes.append(i[-1])
    
len(listTes)


# In[ ]:


columns = ['Input1', 'Input2', 'Input3', 'Input4', 'Input5', 'Input6', 'Input7', 'Input8', 'Input9', 'Input10', 'Input11', 'Input12', 'Input13', 'Input14', 'Input15', 'Input16', 'digit']


# In[ ]:


df_tra = pd.read_csv('../Datasets/pendigits/pendigits.tra', delimiter = ',', names=columns , header=None)
df_tra.head()


# In[ ]:


df_tra_w = df_tra.copy()
df_tra_w['writer'] = listOrigTra
df_tra_w.head()


# In[ ]:


df_tes = pd.read_csv('../Datasets/pendigits/pendigits.tes', delimiter = ',', names=columns , header=None)
df_tes.head()


# In[ ]:


df_tes.to_csv('TEST.csv')


# In[ ]:


df_tes_w = df_tes.copy()
df_tes_w['writer'] = listOrigTes
df_tes_w.head()


# In[ ]:


df = pd.concat([df_tra_w,df_tes_w])
df.head()


# In[ ]:


d0 = df_tra_w.loc[df_tra_w.digit==0].shape[0]
d1 = df_tra_w.loc[df_tra_w.digit==1].shape[0]
d2 = df_tra_w.loc[df_tra_w.digit==2].shape[0]
d3 = df_tra_w.loc[df_tra_w.digit==3].shape[0]
d4 = df_tra_w.loc[df_tra_w.digit==4].shape[0]
d5 = df_tra_w.loc[df_tra_w.digit==5].shape[0]
d6 = df_tra_w.loc[df_tra_w.digit==6].shape[0]
d7 = df_tra_w.loc[df_tra_w.digit==7].shape[0]
d8 = df_tra_w.loc[df_tra_w.digit==8].shape[0]
d9 = df_tra_w.loc[df_tra_w.digit==9].shape[0]

print(d0)
print(d1)
print(d2)
print(d3)
print(d4)
print(d5)
print(d6)
print(d7)
print(d8)
print(d9)

D={
        0:d0,
        1:d1,
        2:d2,
        3:d3,
        4:d4,
        5:d5,
        6:d6,
        7:d7,
        8:d8,
        9:d9,
    
}


# In[ ]:


pickle.dump(D,open('COUNT','wb'))


# In[ ]:


def getClients(data):
    arr = []
    no_of_writers = len(set(data['writer']))
    for i in range(no_of_writers):
        arr.append([])
    for index, row in data.iterrows():
        rowA = row.to_numpy()
        writer = int(rowA[-1])
        data = rowA[:17]
        arr[writer-1].append(data)
    return arr


# In[ ]:


def get_batches(HOLDER):
    federated_data = []
    for i in HOLDER:
        client = []
        for index in range(len(i)//20):
            X= []
            Y= []
            for elements in i[index*20:index*20+20]:
                X.append(np.array([np.float32(elements[0]),
                                   np.float32(elements[1]),
                                   np.float32(elements[2]),
                                   np.float32(elements[3]),
                                   np.float32(elements[4]),
                                   np.float32(elements[5]),
                                   np.float32(elements[6]),
                                   np.float32(elements[7]),
                                   np.float32(elements[8]),
                                   np.float32(elements[9]),
                                   np.float32(elements[10]),
                                   np.float32(elements[11]),
                                   np.float32(elements[12]),
                                   np.float32(elements[13]),
                                   np.float32(elements[14]),
                                   np.float32(elements[15])]))
                Y.append(np.array(np.int32(elements[-1])))
            client.append({
                'x':np.array(X),
                'y':np.array(Y)
            })
        federated_data.append(client)
    return federated_data


# In[ ]:


# federated_train_data = getClients(df_tra_w)
# federated_test_data = getClients(df_tes_w)


# In[ ]:


# federated_train_data =  get_batches(train)
# federated_validation_data =  get_batches(validation)
# federated_test_data =  get_batches(test)


# In[ ]:


def getModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10,activation="sigmoid")
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model


# In[ ]:


import numpy as np
from collections import Counter

def getweight(df):
    LBL = Counter(df.digit)
    maen_arr = np.mean(list(map(lambda x:x[1]/D[x[0]],list(LBL.items()))))
    return maen_arr


# In[ ]:


writer_data = df_tra_w.groupby(df_tra_w.writer)


# In[ ]:


import pickle
d ={}
for i in range(1,31):
    d[i]=writer_data.get_group('{}'.format(i)).sample(50)


# In[ ]:


pickle.dump(d,open('R1','wb'))


# In[ ]:


X = []
for i in range(20):
    d = {}
    for i in range(1,31):
        d[i]=writer_data.get_group('{}'.format(i)).sample(50)
    X.append(d)


# In[ ]:


pickle.dump(X,open('Rs','wb'))

