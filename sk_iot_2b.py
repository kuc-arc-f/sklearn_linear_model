#sk_iot_2
# encoding: utf-8
# 2018/05/18 : time add version.

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import time
import datetime as dt

#
def conv_tm2float(nDim ):
	ret=[]
	for item in nDim:
		ret.append(item.total_seconds())
	return ret

#
rdDim = pd.read_csv("sensors.csv", names=('id', 'temp', 'time') )
#rdDim = rdDim(index_col='time')
#rdDim = rdDim( parse_dates ='time')
rdDim['time_2'] = pd.to_datetime(rdDim['time'])
rdDim['time_2']

#rdDim.info()
#rdDim.describe()

# Y
Y = rdDim["temp"]
Y = np.array(Y, dtype = np.float32).reshape(len(Y) ,1)
# X
#xDim =np.arange(len(Y ))
xAxis= np.array(rdDim['time_2']).reshape(len(rdDim['time_2'] ) ,1)
xAxis

min = rdDim['time_2'].min()
rdDim['diff'] = rdDim['time_2'] -min
#print(rdDim['diff'][0])
diff = conv_tm2float(rdDim['diff'] )
#print(diff )
xDim =np.array(diff )
print(xDim)
#quit()

X = np.array(xDim, dtype = np.float32).reshape(len(xDim ) ,1)
#quit()

print ("start...")
start_time = time.time()

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

interval = int(time.time() - start_time)
print ("実行時間: {}sec".format(interval) )

# plot表示
#plt.plot(Y ,label = "temp")
#plt.plot(pred, label = "predict" )
plt.title("IoT-data predict ")
plt.plot(xAxis , Y    ,label = "temp")
plt.plot(xAxis , pred , label = "predict" )
plt.legend()
plt.grid(True)
plt.xticks(rotation=45 )
plt.show()
