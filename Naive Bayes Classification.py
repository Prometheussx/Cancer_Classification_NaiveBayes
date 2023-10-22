# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:23:34 2023

@author: erdem
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% Data İnclaoud
data = pd.read_csv("Cancer_Data.csv")

#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


#%%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

#scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color = "red",label="Malignant",alpha = 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color = "green",label="Benign",alpha = 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

#%%
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]
y= data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%%
#normailization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% 
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

#%%
# Naive Bayes Algoritma

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

print("Pirnt accuracy of NBC algo",nb.score(x_test,y_test))