# django libraries
from django.http import HttpResponse
from django.shortcuts import render
from django.templatetags.static import static
#from . import models
from home.forms import ImageForm
from pathlib import Path
import os

# libraries for image captioning
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as IM
from englisttohindi.englisttohindi import EngtoHindi

# function to generate caption
def generateCaption(imgPath):
    BASE_DIR = Path(__file__).resolve().parent.parent
    #Import resnet model for feature extraction
    extract_model = ResNet50(include_top=True)
    last = extract_model.layers[-2].output
    image_features_extract_model = Model(inputs = extract_model.input,outputs = last)

    #Import the Pre-trained CNN-RNN(GRU) model
    #modulePath = os.path.dirname(__file__)
    trainedModelURL = BASE_DIR / 'home' / 'nextModel6.h5'
    #trainedModelURL = os.path.join(modulePath,'nextModel6.h5')
    #trainedModelURL = 'https://drive.google.com/file/d/1-2lCrODWbR1uV1igoGbHowm4HC0uhNOJ/view?usp=sharing' 
    reconstructed_model = load_model(trainedModelURL)

    #Import tokenizer from pickle file
    tokenizerFileURL = BASE_DIR / 'home' / 'tokenizer.pickle'
    #tokenizerFileURL = os.path.join(modulePath,'tokenizer.pickle')
    with open(tokenizerFileURL, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    #State vocab size, max length(max tokens)
    vocab_size = len(tokenizer.word_index) 
    max_tokens = 73
    START_TOKEN = '<start> '
    END_TOKEN = ' <end>'
    token_start = tokenizer.word_index[START_TOKEN.strip()]
    token_end = tokenizer.word_index[END_TOKEN.strip()]

    #Presprocess the image
    image_path = imgPath 
    img = IM.load_img(imgPath, target_size=(224, 224))
    x = IM.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image = tf.keras.applications.resnet.preprocess_input(x)

    #Use the prediction coding,extarct image features from ResNet50 model and RNN features from ------>
    #Reconstructed model, use that model to predict
    encoder_input = image_features_extract_model.predict(image)
    encoder_input = tf.reshape(encoder_input,
                                (2048, ))
    encoder_input = np.expand_dims(encoder_input, axis=0)
    shape = (1, max_tokens)
    decoder_input = np.zeros(shape=shape, dtype=np.int)
    token_id = token_start

    output=[]
    count_tokens = 0
    while token_id != token_end and count_tokens < max_tokens: 
        decoder_input[0, count_tokens] = token_id
        input_data ={'encoder_input':encoder_input ,'decoder_input': decoder_input} 
        predict = reconstructed_model.predict(input_data)
        token_id = np.argmax(predict[0, count_tokens, :])
        output.append(token_id)
        count_tokens += 1
    
    #Print prediction 
    Caption = str(tokenizer.sequences_to_texts([output])[0])
    Caption = ''.join(Caption).split()
    Caption.pop()
    sentence = " ".join(Caption)  
    return sentence

# Function to render home page
def home(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            #print(img_obj)
            pathToImage = img_obj.image.url #this will change later
            caption = generateCaption(pathToImage)
            res = EngtoHindi(caption)
            return render(request,'index.html',{'form': form,'engCaption':caption, 'hindiCaption': res.convert})
    if request.method == 'GET':
        form = ImageForm()
        return render(request,'index.html',{'form': form})
    
    return HttpResponse('Error', content_type='text/plain')
    
            


