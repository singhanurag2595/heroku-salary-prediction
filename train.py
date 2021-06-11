import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

data=pd.read_csv('hiring.csv')

print(data)
print(data.isnull().sum())
print(len(data))

data.columns=['experience',  'test_score',  'interview_score', 'salary']

data.experience.fillna(0 , inplace = True)
data.test_score.fillna(data.test_score.mean() , inplace = True)

print(data)

X = data.iloc[: , :3]

def convert_to_int(word):
    word_dic = {'one':1 , 'two':2 , "three":3 , "four":4 , 'five':5 , 'six':6 , "seven":7 , 'eight':8 , 'nine':9 , 'ten':10 , 'eleven':11 , 0:0}
    return word_dic[word]

X['experience'] = X.experience.apply(lambda x : convert_to_int(x))

y = data.iloc[: , -1]

regressor = LinearRegression()
regressor.fit(X,y)
print('Model Training is done')

joblib.dump(regressor , 'hiring_model.pkl',)

print(regressor.predict([[1 , 8,9]]))