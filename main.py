import random
import numpy as np
import pickle
from model import siamese
import tensorflow as tf
from tensorflow.keras.models import model_from_json
def dataloader(dataset,batch_size=32):
    
    pairs=[np.zeros((batch_size,100,7)) for _ in range(2)]
    labels=np.zeros((batch_size,1))
   
    label_subset = random.choices([i for i in range(len(dataset))], k=batch_size)
    for i,index in enumerate(label_subset):
  
        idx_1 = random.randint(0, len(dataset[index])-1)

        pairs[0][i,:,:] = dataset[index][idx_1]


        if i >= batch_size // 2:
            idx_2 = random.randint(0, len(dataset[index])-1)
        
            pairs[1][i,:,:] = dataset[index][idx_2]
            labels[i]=1

        else:
            #idx_2 = random.randint(0, len(dataset[index])-1)
            index1 = (index + random.randint(1,len(dataset)-1)) % len(dataset)
            idx_2 = random.randint(0, len(dataset[index1])-1)
            pairs[1][i,:,:] = dataset[index1][idx_2]
            labels[i]=0


   
    return pairs,labels

def generate(dataset,batch_size=32):

    while True:
        pairs, targets = dataloader(dataset,batch_size)
        yield (pairs, targets)





def train(train_set,dev_set,model):

    Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'siemese', monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True, mode='max', save_freq='epoch',
    options=None
    )

    model.fit(
        generate(train_set),
        validation_data=generate(dev_set),
        epochs=200,verbose=1,
        steps_per_epoch=1000,
        validation_steps=10,
        callbacks = Checkpoint
        )


    siemese_json=model.to_json()
    with open("siemese.json", "w") as json_file:
        json_file.write(siemese_json)

  

def Eval(test_set,batch_size=32):
    with open('siemese.json','r') as jason_file:
        model_json = jason_file.read()
    model = model_from_json(model_json)
    
    model.load_weights('siemese')
    model.save_weights('siemese.h5')
    num_correct = 0
    for _ in range(10):
        
        
        #num_correct = 0
        pairs,label = dataloader(test_set,32)
        
      
        test_preds = model.predict(pairs)
          
        test_preds = tf.round(test_preds).numpy()
    
        num_correct = num_correct+(test_preds == label).sum()

    accuracy = num_correct / (batch_size*10)

    print('test on {} instances, {} are correct, accuracy: {}'.format(batch_size*10,num_correct,accuracy))





if __name__=='__main__':
    
    with open('train.pkl','rb') as f:

        train_set = pickle.load(f)


    
    with open('dev.pkl','rb') as f1:

        dev_set = pickle.load(f1)

    with open('test.pkl','rb') as f2:

        test_set = pickle.load(f2)

    #model = siamese((100,7))
    #train(train_set,dev_set,model)

    Eval(test_set)