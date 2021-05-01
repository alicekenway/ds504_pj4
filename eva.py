import pickle
import numpy as np
import pandas as pd
import random
from collections import defaultdict

import random


from model import siamese
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from preprocessing import process_data

with open('validate_set.pkl','rb') as f:

    valid_set = pickle.load(f)



with open('validate_label.pkl','rb') as f:

    labels = pickle.load(f)



def pred(model,pair,label):
    lefts = process_data(pair[0],'test')
    rights = process_data(pair[1],'test')

    l = min(len(lefts),len(rights))
    lefts = lefts[:l]
    rights = rights[:l]

    L = np.zeros((l,100,7))
    R = np.zeros((l,100,7))
    for i in range(l):
        L[i,:,:] = lefts[i]
        R[i,:,:] = rights[i]
    #print(label)
    label = np.transpose(np.array([[label]*l]))
    #print(label)
    prd = model.predict([L,R])
    
    prd = tf.round(prd).numpy()
    #print((prd.sum(),l))
    #print(label[0])
    if (prd==label).sum() >l//2:
        return 1
    else:
        return 0

with open('siemese.json','r') as jason_file:
    model_json = jason_file.read()
model = model_from_json(model_json)
model.load_weights('siemese.h5')

sum_ = 0
for i in range(len(valid_set)):
    sum_+=pred(model,valid_set[i],labels[i])


print(sum_/len(valid_set))