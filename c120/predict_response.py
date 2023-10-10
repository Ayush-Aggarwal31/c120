#Text Data Preprocessing Lib
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('./Chatbot.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

def preprocess_user_input(userinput):
    inputwordtoken1=nltk.word_tokenize(userinput)
    inputwordtoken2=get_stem_words(inputwordtoken1,ignore_words)
    inputwordtoken2=sorted(list(set(inputwordtoken2)))
    bag=[]
    bagofwords=[]
    for word in words:
        if(word in inputwordtoken2):
            bagofwords.append(1)
        else:
            bagofwords.append(0)
    bag.append(bagofwords)
    return np.array(bag)
    
def bot_class(userinput):
    inp=preprocess_user_input(userinput)
    prediction=model.predict(inp)
    predictedclass=np.argmax(prediction[0])
    return predictedclass
def bot_response(userinput):
    predictedclasslabel=bot_class(userinput)
    predictedclass=classes[predictedclasslabel]
    for i in intents["intents"]:
        if(i["tag"]==predictedclass):
            bot_response=random.choice(i["responses"])
            return bot_response
        


print("Hi, this is Alexa! How can 1 assist you?")

while True:
    user_input = input("Type your message here...")
    print("User Input : ",user_input)

    response = bot_response(user_input)
    print("Bot Response : ",response)