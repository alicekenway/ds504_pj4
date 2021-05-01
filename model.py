import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

def siamese(input_shape):

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    model = Sequential()
    
    model.add(CuDNNLSTM(100,return_sequences=True,input_shape=input_shape))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(CuDNNLSTM(100))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128,activation=tf.nn.sigmoid))
 

    net1 = model(input1)
    net2 = model(input2)
    L1_layer = Lambda(lambda x:K.abs(x[0] - x[1]))
    L1_distance = L1_layer([net1, net2])
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[input1,input2],outputs=prediction)
    optimizer = Adam(lr = 0.0003)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    return siamese_net

if __name__=='__main__':
    a = siamese((100,7))
    a.summary()
    plot_model(a, to_file='try.png', show_shapes=True, show_layer_names=True,expand_nested=True)