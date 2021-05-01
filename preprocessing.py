import pickle
import numpy as np
import pandas as pd
import random
from collections import defaultdict


def load_data(filename):
    with open(filename,'rb') as f:
        dataset = pickle.load(f)
        
    return dataset




def process_data(traj,mode ='train'):
    if mode=='train':
        frame = pd.DataFrame(traj, columns =['Plate','longitude', 'latitude','Second_since_midnight','status','time'])
        frame = frame.drop(['Plate'],axis=1)
    else:
        frame = pd.DataFrame(traj, columns =['longitude', 'latitude','Second_since_midnight','status','time'])
    frame = frame.drop(['Second_since_midnight'],axis=1)
    frame['time'] = pd.to_datetime(frame['time'])
    

    f1 = frame['longitude']<180 

    frame1 = frame[f1]
    f2 = frame1['latitude']<180
    frame2 = frame1[f2]
    frame = frame2
    del frame1
    del frame2

    frame['second'] = frame['time'].dt.hour * 3600 + \
             frame['time'].dt.minute * 60 + \
             frame['time'].dt.second

    frame['day'] = frame['time'].dt.day
    frame['day_sin'] = np.sin(2 * np.pi * (frame['day']-1)/30)
    frame['day_cos'] = np.cos(2 * np.pi * (frame['day']-1)/30)
    frame['second_sin'] = np.sin(2 * np.pi * (frame['second']-1)/86399)
    frame['second_cos'] = np.cos(2 * np.pi * (frame['second']-1)/86399)
    frame = frame.loc[:,~frame.columns.isin(['second','day'])]
    frame['longitude'] = frame['longitude']/180
    frame['latitude'] = frame['latitude']/180    
    #print(frame.head())
    #label = frame['Plate'][0]
    
    partition = frame.sort_values(['time']).drop(['time'],axis=1).to_numpy()

    window_size=100
    full_list = []
    length = partition.shape[0]
    if length<window_size:
        last = np.expand_dims(partition[-1],axis=0)
        last = np.repeat(last,window_size-length,axis=0)
        partition = np.vstack((partition,last))
        length = window_size
    
    if mode=='train':
        for i in range(0,length-window_size+1,20):
            full_list.append(partition[i:i+window_size,:])    
    else:
        for i in range(0,length-window_size+1,100):
            full_list.append(partition[i:i+window_size,:])    

    #full_list = np.array(full_list,dtype=np.float)
    return full_list


def dataset_making(dataset):

    random.seed(1)
    full_list = []
    for i in range(len(dataset)):
        

        L = []
        for j in range(len(dataset[i])):
            traj =  process_data(dataset[i][j])
            L.extend(traj)
        full_list.append(L)
    random.shuffle(full_list)
    train = full_list[:int(len(full_list)*0.9)]
    dev = full_list[int(len(full_list)*0.9) : int(len(full_list)*0.95)]
    test = full_list[int(len(full_list)*0.95):]
                
    return train,dev,test

if __name__=='__main__':
    dataset = load_data('project_4_train.pkl')
    train,dev,test = dataset_making(dataset)

    with open('train.pkl','wb') as f1:
        pickle.dump(train,f1)
        
        
    with open('dev.pkl','wb') as f2:

        pickle.dump(dev,f2)
        
    with open('test.pkl','wb') as f3:
     
        pickle.dump(test,f3)



