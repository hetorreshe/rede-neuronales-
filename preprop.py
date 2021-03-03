import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/train_gamma.csv")
df=df.iloc[:,1:]

dftest=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/test_gamma.csv")
dftest=dftest.iloc[:,1:]
def preprocess(df, history_size):
    data=[]
    X=df.drop("target", axis=1)
    y=df["target"].tolist()
    for i in range(history_size, len(X.index)):
        data.append(np.array(X.loc[i-history_size:i]).reshape(1,len(X.columns)*(history_size+1)).tolist()[0])
    y=y[history_size:]
    return np.array(data), np.array(y)

X,y= preprocess(df, 2)
X

X_test, y_test=preprocess(dftest, 2)


import tensorflow as tf
from tensorflow import keras

X.shape
X_test.shape
y.shape
y_test.shape


variables=X.shape[1]
tf.random.set_seed(1997)
model=keras.Sequential()
model.add(keras.layers.Dense(64, activation="elu",input_shape=(variables,)))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(2, activation="softmax"))
model.summary()


model.compile(loss="sparse_categorical_crossentropy",
              metrics=["accuracy"],
              optimizer="adam")


history=model.fit(X, y,
                  epochs=50,
                  batch_size=32,
                  validation_data=(X_test, y_test))

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="test")
plt.legend()

dfpred1=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/gammapred1.csv")
dfpred1=dfpred1.iloc[:,1:]

X_pred1, y_pred1=preprocess(dfpred1, 2)

predicts=model.predict(X_pred1)
predictions=[]
for i in predicts:
    if i[0]>i[1]:
        predictions.append(0)
    elif i[0]<i[1]:
        predictions.append(1)

predictions=np.array(predictions)
sum(predictions)
sum(y_pred1)
bools=predictions==y_pred1
sum(bools)


dfpred2=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/gammapred2.csv")
dfpred2=dfpred2.iloc[:,1:]

X_pred2, y_pred2=preprocess(dfpred2, 2)

predicts2=model.predict(X_pred2)
predictions2=[]
for i in predicts2:
    if i[0]>i[1]:
        predictions2.append(0)
    elif i[0]<i[1]:
        predictions2.append(1)

predictions2=np.array(predictions2)
sum(predictions2)
sum(y_pred2)
bools=predictions2==y_pred2
sum(bools)


dfpred3=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/gammapred3.csv")
dfpred3=dfpred3.iloc[:,1:]

X_pred3, y_pred3=preprocess(dfpred3, 2)

predicts3=model.predict(X_pred3)
predictions3=[]
for i in predicts3:
    if i[0]>i[1]:
        predictions3.append(0)
    elif i[0]<i[1]:
        predictions3.append(1)

predictions3=np.array(predictions3)
sum(predictions3)
sum(y_pred3)
bools=predictions3==y_pred3
sum(bools)


dfpred4=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/gammapred4.csv")
dfpred4=dfpred4.iloc[:,1:]

X_pred4, y_pred4=preprocess(dfpred4, 2)

predicts4=model.predict(X_pred4)
predictions4=[]
for i in predicts4:
    if i[0]>i[1]:
        predictions4.append(0)
    elif i[0]<i[1]:
        predictions4.append(1)

predictions4=np.array(predictions4)
sum(predictions4)
sum(y_pred4)
bools=predictions4==y_pred4
sum(bools)


dfpred5=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/gammapred5.csv")
dfpred5=dfpred5.iloc[:,1:]

X_pred5, y_pred5=preprocess(dfpred5, 2)

predicts5=model.predict(X_pred5)
predictions5=[]
for i in predicts5:
    if i[0]>i[1]:
        predictions5.append(0)
    elif i[0]<i[1]:
        predictions5.append(1)

predictions5=np.array(predictions5)
sum(predictions5)
sum(y_pred5)
bools=predictions5==y_pred5
sum(bools)


dfpred6=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/pred6.csv")
dfpred6=dfpred6.iloc[:,1:]

X_pred6, y_pred6=preprocess(dfpred6, 2)

predicts6=model.predict(X_pred6)
predictions6=[]
for i in predicts6:
    if i[0]>i[1]:
        predictions6.append(0)
    elif i[0]<i[1]:
        predictions6.append(1)

predictions6=np.array(predictions6)
sum(predictions6)
sum(y_pred6)
bools=predictions6==y_pred6
sum(bools)


dfpred7=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/pred7.csv")
dfpred7=dfpred7.iloc[:,1:]

X_pred7, y_pred7=preprocess(dfpred7, 2)

predicts7=model.predict(X_pred7)
predictions7=[]
for i in predicts7:
    if i[0]>i[1]:
        predictions7.append(0)
    elif i[0]<i[1]:
        predictions7.append(1)

predictions7=np.array(predictions7)
sum(predictions7)
sum(y_pred7)
bools=predictions7==y_pred7
sum(bools)


dfpred8=pd.read_csv("D:/Scripts/R Scripts/Control de Calidad/LSTM/pred8.csv")
dfpred8=dfpred8.iloc[:,1:]

X_pred8, y_pred8=preprocess(dfpred8, 2)

predicts8=model.predict(X_pred8)
predictions8=[]
for i in predicts8:
    if i[0]>i[1]:
        predictions8.append(0)
    elif i[0]<i[1]:
        predictions8.append(1)

predictions8=np.array(predictions8)
sum(predictions8)
sum(y_pred8)
bools=predictions8==y_pred8
sum(bools)
