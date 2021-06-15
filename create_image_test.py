import numpy as np
from PIL import Image
import pandas as pd
import os

def make_image(array):
    data = Image.fromarray(array) 
    return data


df = pd.read_csv("data.csv")
df.drop('Date', axis=1, inplace=True)
np_array = df.to_numpy()


root = 'D:/Programing/Python/Image procesing in Finance/Data/TSLA/Images/' # root data path



for i in range(0,np.shape(np_array)[0]-14):
    img = np_array[i:i+15]
    img.reshape((15,15,1))
    img = np.round(img*255)
    img = img.astype('int')
    img = make_image(img)
    
    img.save(root+'/img' +str(i)+'.png')
