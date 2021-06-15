import numpy as np
from PIL import Image
import pandas as pd
import os

def make_image(array):
    data = Image.fromarray(array) 
    return data

def get_labels(data):
    data_array = data["Close"].to_numpy()
    labels = np.array([0 for i in range(len(data_array))]).astype("uint8")
    for i in range(0,len(data_array) - 20):
        window = data_array[i:i+15]
        minindex = np.argmin(window)
        maxindex = np.argmax(window)
        labels[i + minindex] = 1
        labels[i + maxindex] = 2
    return labels

df = pd.read_csv("data.csv")
df.drop('Date', axis=1, inplace=True)
np_array = df.to_numpy()
labels = get_labels(df)


root = 'D:/Programing/Python/Image procesing in Finance/Data/HINDUNILVR/' # root data path
buy = root + 'Buy'
sell = root + 'Sell'
hold = root + 'Hold'

try:
    os.mkdir(buy)
except:
    print("Directory alredy exist or OSError")

try:  
    os.mkdir(sell)
except:
    print("Directory alredy exist or OSError")

try:    
    os.mkdir(hold)
except:
    print("Directory alredy exist or OSError")



for i in range(0,np.shape(np_array)[0]-14):
    img = np_array[i:i+15]
    img.reshape((15,15,1))
    img = np.round(img*255)
    img = img.astype('int')
    img = make_image(img)
    
    if (labels[i] == 0):
        img.save(hold+'/img' +str(i)+'.png')
    if (labels[i] == 1):    
        img.save(buy+'/img' +str(i)+'.png')
    if (labels[i] == 2):
        img.save(sell+'/img' +str(i)+'.png')