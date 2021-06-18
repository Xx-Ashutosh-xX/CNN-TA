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


root = 'D:/Programing/Python/Image procesing in Finance/Data/TTM/Images/' # root data path



for i in range(0,np.shape(np_array)[0]-14):
    img = np_array[i:i+15]
    img = np.transpose(img)
    for j in range (0,15):
        img[j] = (img[j]- np.min(img[j]))/(np.max(img[j]))
        img[j] = np.abs(img[j])
    img = np.round(img*255)
    img.reshape((15,15,1))
    img = img.astype('int')
    img = make_image(img)

    img.save(root+'img' +str(i)+'.png')