import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import mlflow
import mlflow.keras
#reading the data
train_data=pd.read_csv('train.csv')
train_data.head(5)
#mlflow.set_tracking_uri("http://10.42.204.118:8000")
#exp_id=mlflow.create_experiment("sequential CNN")

X = train_data.iloc[:,:20].values
y = train_data.iloc[:,20:21].values

sc = StandardScaler()
X = sc.fit_transform(X)
ohe = OneHotEncoder()
#encoding classes into binary values
y = ohe.fit_transform(y).toarray()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
with mlflow.start_run():
    #using autologging for logging parameters,metrics etc.
    mlflow.keras.autolog()
    #Building Neural network
    model = Sequential()
    model.add(Dense(16, input_dim=20, activation='relu'))
    model.add(Dense(12, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=32)

    y_pred = model.predict(X_test)

    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
        a = accuracy_score(pred,test)
        print('Accuracy is:', a*100)
