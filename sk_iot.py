# encoding: utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#
#rdDim = pd.read_csv("sensors_f1.csv" , names=('id', 'temp', 'time'))
rdDim = pd.read_csv("sensors.csv", names=('id', 'temp', 'time') )
 
# Y
Y = rdDim["temp"]
Y = np.array(Y, dtype = np.float32).reshape(len(Y) ,1)
# X
xDim =np.arange(len(Y ))
X = np.array(xDim, dtype = np.float32).reshape(len(xDim ) ,1)
#print(len(X) )
#print(X[:20] )
#quit()

# 予測モデルを作成
clf = linear_model.LinearRegression()
clf.fit(X, Y)
# 回帰係数
print(clf.coef_)
# 切片 (誤差)
print(clf.intercept_)
# 決定係数
print(clf.score(X, Y))

#predict
pred = clf.predict(X)
 
# plot表示
plt.plot(Y ,label = "temp")
plt.plot(pred, label = "predict" )
plt.legend()
plt.grid(True)
plt.show()
