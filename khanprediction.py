import numpy as np
import pandas as pd
data = pd.read_csv('./ohe_data_reduce_cat_class (1).csv')


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
data.columns = data.columns.str.strip()
data[['total_sqft_int','price_per_sqft']] = sc.fit_transform(data[['total_sqft_int','price_per_sqft']])
x = data.drop('price',axis = 'columns')
y = data.price
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 52)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)
print(lr.score(xtest, ytest))


import pickle
path = "./hpp.pickle"
pickle_out = open(path,"wb")
pickle.dump(lr,pickle_out)
pickle_out.close()


# for predicting take the array of 38 columns mentioned in the dataset
# convert it in to the array of dimention 2 
# xtest = np.array(xtest)
# a = xtest[0]  # 1 dimension array of 38 input
# a = np.reshape(a,(1,-1)) # here creating a 2 dimentional array
# khan .predict(a)

