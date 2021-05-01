import pickle
import numpy as np
import random
from model import siamese
import tensorflow as tf
from preprocessing import process_data as process

from tensorflow.keras.models import model_from_json
validate = pickle.load(open("validate_set.pkl",'rb'))
label = pickle.load(open("validate_label.pkl",'rb'))

def load_model():
    '''
    Load your model
    '''

    model = siamese((100,7))
    model.load_weights('siemese.h5')
    return model


def process_data(traj_1, traj_2):
    """
    Input:
        Traj: a list of list, contains one trajectory for one driver 
        example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
            [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
        Data: any format that can be consumed by your model.
    
    """
    lefts = process(traj_1,'test')
    rights = process(traj_2,'test')
    
    l = min(len(lefts),len(rights))
    lefts = lefts[:l]
    rights = rights[:l]

    L = np.zeros((l,100,7))
    R = np.zeros((l,100,7))


    for i in range(l):
        L[i,:,:] = lefts[i]
        R[i,:,:] = rights[i]

    
    return [L,R]

def run(data, model):
    """
    
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    
    """
    prds = model.predict(data)
    #print(data[0].shape[0])
    #print(prds)
    prds = tf.round(prds).numpy()
    #print(prds)
    #print(prd.sum())
    if prds.sum() >data[0].shape[0]//2:
        return 1
    else:
        return 0

    


if __name__=='__main__':
    model = load_model()
    s = 0
    for d, l in zip(validate,label):
        data = process_data(d[0],d[1])
        prd = run(data,model)
        #print(prd)
        if l==prd:
            s+=1
    print(s/len(validate))
    print(len(validate))