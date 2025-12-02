import numpy as np
import pandas as pd   
import os
from pathlib import Path
import glob
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import nltk
import cv2
import matplotlib.pyplot as plt
import random
from data_utils import MyDataLoader

#Check GPU is available for training or not Or whether the tensorflow version can utilize gpu 
physical_devices = tf.config.list_physical_devices('GPU') 
print("Number of GPUs :", len(physical_devices)) 
print("Tensorflow GPU :",tf.test.is_built_with_cuda())
if len(physical_devices)>0:
    device="/GPU:1"
else:
    device="/CPU:0"

#Loading the dataset
data_loader = MyDataLoader(
    image_path="/home/jnlp/minhnt/AdvML/custom_dataset",
    data_path="/home/jnlp/minhnt/AdvML/custom_dataset/train_split.csv",
    split="train_split"
)
train_dataframe, train_image_dir = data_loader.data
train_dataframe['path'] = train_image_dir + '/' + train_dataframe['file']
data_loader = MyDataLoader(
    image_path="/home/jnlp/minhnt/AdvML/custom_dataset",
    data_path="/home/jnlp/minhnt/AdvML/custom_dataset/dev_split.csv",
    split="dev"
)
val_dataframe, val_image_dir = data_loader.data
val_dataframe['path'] = val_image_dir + '/' + val_dataframe['file']

# text encoder
vocab_set=set()#set object used to store the vocabulary

tokenizer = tfds.deprecated.text.Tokenizer()

for i in val_dataframe['question']:
    vocab_set.update(tokenizer.tokenize(i))
for i in train_dataframe['question']:
    vocab_set.update(tokenizer.tokenize(i))
for i in val_dataframe['answer']:
    vocab_set.update(tokenizer.tokenize(i))
for i in train_dataframe['answer']:
    vocab_set.update(tokenizer.tokenize(i))
#
#Creating an Encoder and a Function to preprocess the text data during the training and inference    
    
encoder=tfds.deprecated.text.TokenTextEncoder(vocab_set)
index=14
print("Testing the Encoder with sample questions - \n ")
example_text=encoder.encode(train_dataframe['question'][index])
print("Original Text = "+train_dataframe['question'][index])
print("After Encoding = "+str(example_text))

# imgtest = tf.io.read_file(val_dataframe.iloc[0]['path'])
# print(val_dataframe.iloc[0]['path'])

IMG_SIZE=(200,200)
# IMG_SIZE=(480,320)
BATCH_SIZE=50

# Function that uses the encoder created to encode the input question and answer string
def encode_fn(text):
    return np.array(encoder.encode(text.numpy()))


#Function to load and decode the image from the file paths in the dataframe and use the encoder function
def preprocess(ip,ans):
    img,ques=ip#ip is a list containing image paths and questions
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=3)
    # quantos canais de cores tem 
    img=tf.image.resize(img,IMG_SIZE)
    img=tf.math.divide(img, 255)# 
    #The question string is converted to encoded list with fixed size of 50 with padding with 0 value
    ques=tf.py_function(encode_fn,inp=[ques],Tout=tf.int32)
    paddings = [[0, 50-tf.shape(ques)[0]]]
    ques = tf.pad(ques, paddings, 'CONSTANT', constant_values=0)
    ques.set_shape([50])#Explicit shape must be defined in order to create the Input pipeline
    
    #The Answer is also encoded 
    ans=tf.py_function(encode_fn,inp=[ans],Tout=tf.int32)
    ans.set_shape([1])
    
    return (img,ques),ans
    
def create_pipeline(dataframe):
    raw_df=tf.data.Dataset.from_tensor_slices(((dataframe['path'],dataframe['question']),dataframe['answer']))
    df=raw_df.map(preprocess)#Preprocessing function is applied to the dataset
    df=df.batch(BATCH_SIZE)#The dataset is batched
    return df

#The training and validation Dataset objects are created
train_dataset=create_pipeline(train_dataframe)
validation_dataset=create_pipeline(val_dataframe)


#Creating the CNN model for image processing
CNN_Input=tf.keras.layers.Input(shape=(200,200,3),name='image_input')

mobilenetv2=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(200,200,3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=CNN_Input)

CNN_model=tf.keras.models.Sequential()
CNN_model.add(CNN_Input)
CNN_model.add(mobilenetv2)
CNN_model.add(tf.keras.layers.GlobalAveragePooling2D())

#Creating the RNN model for text processing
RNN_model=tf.keras.models.Sequential()

RNN_Input=tf.keras.layers.Input(shape=(50,),name='text_input')
RNN_model.add(RNN_Input)
RNN_model.add(tf.keras.layers.Embedding (len(vocab_set)+1,256))
RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,stateful=False,return_sequences=True,recurrent_initializer='glorot_uniform')))
RNN_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,stateful=False,return_sequences=False,recurrent_initializer='glorot_uniform')))

print(CNN_model.summary())
print(RNN_model.summary())

concat=tf.keras.layers.concatenate([CNN_model.get_layer('global_average_pooling2d').output,RNN_model.get_layer('bidirectional_2').output])
dense_out=tf.keras.layers.Dense(len(vocab_set)+1,activation='softmax',name='output')(concat)

model = tf.keras.Model(inputs=[CNN_Input,RNN_Input], outputs=dense_out)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
print(model.summary())


def scheduler(epoch):
  if epoch < 1:
    return 0.001
  else:
    # return 0.001 * tf.math.exp(0.1 * (1 - epoch))
    return float(0.001 * tf.math.exp(0.1))

LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)
csv_callback=tf.keras.callbacks.CSVLogger(
    "Training Parameters.csv", separator=',', append=False
)
teste =1

#create a checkpoint to save the training

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "/home/jnlp/minhnt/AdvML/dl_ckpt/cp-{epoch:04d}.weights.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)
epoch = 0

epoch = 0


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath='{epoch:02d}.weights.h5', 
    verbose=1, 
    save_weights_only=True,
    save_freq=1000*BATCH_SIZE)



# def FormatarEndereco(epoch):
#    epoch = epoch+1
#    return "training_2/cp-"+ str(epoch) + ".ckpt"

# model.save_weights(FormatarEndereco(epoch))

model.save_weights(checkpoint_path.format(epoch=0,val_loss=0))
                   
with tf.device(device):
   history =  model.fit(train_dataset,
              validation_data=validation_dataset,
              callbacks=[csv_callback,LRS,cp_callback],
              epochs=10)
   
# load the weights from the h5 checkpoint
# model.load_weights("/home/jnlp/minhnt/AdvML/dl_ckpt/cp-0000.weights.h5")
   
for i in range(5):
    index=i
    # fig,axis=plt.subplots(1,2,figsize=(25, 8))
    im=cv2.imread(val_dataframe.iloc[index]['path'])
    im=cv2.resize(im,(200,200))
    q=val_dataframe.iloc[index]['question']
    q=encoder.encode(q)
    paddings = [[0, 50-tf.shape(q)[0]]]
    q=tf.pad(q, paddings, 'CONSTANT', constant_values=0)
    q=np.array(q)
    im.resize(1,200,200,3)
    q.resize(1,50)
    print(im.shape)
    print(q.shape)
    ans=model.predict([[im],[q]])
    decoded_ans = encoder.decode([np.argmax(ans)])
    print("Question : ", val_dataframe.iloc[index]['question'])
    print("Predicted Answer : "+ decoded_ans)
    print("Actual Answer : "+ val_dataframe.iloc[index]['answer'])
    # question=""
    # flag=0
    # for i,j in enumerate(val_dataframe.iloc[index]['question']):
    #     if (flag==1) and (j==' '):
    #         question+='\n'
    #         flag=0
    #     question+=j
    #     if (i%40==0)and (i!=0):
    #         flag=1
    # axis[0].imshow(im)
    # axis[0].axis('off')
    # axis[0].set_title('Image', fontsize=30)
    # axis[1].text(0.05,0.5,
    #          "Question = {}\n\nPredicted Answer = {}\n\nActual Answer ={}".format(question,encoder.decode([np.argmax(ans)]),val_dataframe.iloc[index]['Answer']),
    #          transform=plt.gca().transAxes,fontsize=19)
    # axis[1].axis('on')
    # axis[1].set_title('Question And Answers', fontsize=30)