import random
import numpy as np
import pickle
from model import siamese
import tensorflow as tf
from tensorflow.keras.models import model_from_json
def dataloader(dataset,shot=64,classes=10,batch_size=64,repea=4):
    
    pairs=[np.zeros((shot*classes,100,7)) for _ in range(2)]
    labels=np.zeros((shot*classes,1))
   
    label_subset = random.choices([i for i in range(len(dataset))], k=classes)
    for j,index in enumerate(label_subset):
        for i in range(shot):
            idx_1 = random.randint(0, len(dataset[index])-1)

            pairs[0][j*shot+i,:,:] = dataset[index][idx_1]


            if i >= shot // 2:
                idx_2 = random.randint(0, len(dataset[index])-1)
            
                pairs[1][j*shot+i,:,:] = dataset[index][idx_1]
                labels[j*shot+i]=1

            else:
                #idx_2 = random.randint(0, len(dataset[index])-1)
                index1 = (index + random.randint(1,len(dataset)-1)) % len(dataset)
                idx_2 = random.randint(0, len(dataset[index1])-1)
                pairs[1][j*shot+i,:,:] = dataset[index1][idx_2]
                labels[j*shot+i]=0
    dataset = tf.data.Dataset.from_tensor_slices(
            (pairs[0].astype(np.float32),pairs[1].astype(np.float32), labels.astype(np.int32))
        )

    dataset = dataset.shuffle(shot*classes).batch(batch_size).repeat(repea)
    return dataset




meta_iters = 150
meta_step_size = 0.25
inner_iters = 4
eval_interval = 5

def train(train_set,dev_set,model,shot=10,classes=64,repea=4):

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003)
    max_dev_acc = 0
    for meta_iter in range(meta_iters):
        model = siamese((100,7))
        cur_meta_step_size = (1 - meta_iter/meta_iters)*meta_step_size

        old_p = model.get_weights()

        dataset = dataloader(train_set,shot,classes)
        
        for pair1,pair2,label in dataset:
            
            with tf.GradientTape() as tape:
                preds = model([pair1,pair2])
                loss = tf.keras.losses.binary_crossentropy(label, preds)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            new_p = model.get_weights()
            for var in range(len(new_p)):
                new_p[var] = old_p[var] + (
                    (new_p[var] - old_p[var]) * cur_meta_step_size
                )

            
        if meta_iter % eval_interval == 0:
            
            accuracies = []
            for set_ in (train_set,dev_set):
                num_correct = 0
                dataset = dataloader(set_,shot,classes)
              
                for pair1,pair2,label in dataset:
                    #print(label.numpy())
                    test_preds = model.predict([pair1,pair2])
                    test_preds = tf.round(test_preds).numpy()
                #test_preds = tf.argmax(test_preds).numpy()
                #print(test_preds)
                    num_correct = num_correct+(test_preds == label.numpy()).sum()

                accuracies.append(num_correct / (shot*classes*repea))

            print('iter: {}  train_acc: {}  dev_acc:{}'.format(meta_iter,accuracies[0],accuracies[1]))
            if accuracies[1]>max_dev_acc:
                siemese_json=model.to_json()
                with open("siemese.json", "w") as json_file:
                    json_file.write(siemese_json)

                model.save_weights('siemese.h5')      
                print('accuracy improve from {} to {}, update'.format(max_dev_acc,accuracies[1]))
                max_dev_acc = accuracies[1]

def Eval(test_set,shot=10,classes=64,repea=1):
    with open('siemese.json','r') as jason_file:
        model_json = jason_file.read()
    model = model_from_json(model_json)

    model.load_weights('siemese.h5')
    num_correct = 0
    for _ in range(10):
        
        
        #num_correct = 0
        dataset = dataloader(test_set,shot,classes,repea=1)
        
        for pair1,pair2,label in dataset:
            #print(label.numpy())
            test_preds = model.predict([pair1,pair2])
            print(test_preds)
            test_preds = tf.round(test_preds).numpy()
        #test_preds = tf.argmax(test_preds).numpy()
        #print(test_preds)
            print(label.numpy())
            num_correct = num_correct+(test_preds == label.numpy()).sum()

    accuracy = num_correct / (shot*classes*repea*10)

    print('test on {} instances, {} are correct, accuracy: {}'.format(shot*classes*repea*10,num_correct,accuracy))





if __name__=='__main__':
    
    with open('train.pkl','rb') as f:

        train_set = pickle.load(f)


    
    with open('dev.pkl','rb') as f1:

        dev_set = pickle.load(f1)

    with open('test.pkl','rb') as f2:

        test_set = pickle.load(f2)

    model = siamese((100,7))
    train(train_set,dev_set,model)

    Eval(test_set,shot=10,classes=64,repea=1)