import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#------------------------ Understand Data -------------------------:

data=pd.read_excel("covid19_Turkey.xlsx")
#print(data.head(10))
#data.plot()
#plt.show()
#print(data.isnull().sum())

#------------------------ Clearing Data -------------------------:

data=data.drop("Country/Region",axis=1)
#print(data.head(10))

#------------------------ Create Model -------------------------:

X=data[["Day"]].values
y=data["Confirmed"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=15)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

model=Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=300)
model.save("Covid19_Turkey.h5")

# ------------------------ Evaluate The Results ---------------------------:

lossData=pd.DataFrame(model.history.history)
#print(lossData.head())
#lossData.plot()
#plt.show()

trainLoss=model.evaluate(X_train,y_train,verbose=0)
testLoss=model.evaluate(X_test,y_test,verbose=0)

#print("Train Loss: ",trainLoss)
#print("Test Loss: ",testLoss)

testPredicts=model.predict(X_test)

predictDF=pd.DataFrame(y_test,columns=["Real Y"])

testPredicts=pd.Series(testPredicts.reshape(59,))

predictDF=pd.concat([predictDF,testPredicts],axis=1)
predictDF.columns=["Real Y","Predict Y"]
#print(predictDF)

newPrecitValue=input("Enter day for predict: ")
newPrecitValue=[[newPrecitValue]]
newPrecitValue=scaler.transform(newPrecitValue)
predict=model.predict(newPrecitValue)

print("Predict: ",int(predict)," confirmeds")