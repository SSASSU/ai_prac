import tensorflow as tf
import pandas as pd
 
csv_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemon_pd = pd.read_csv(csv_path)
lemon_pd.head()

dep = lemon_pd[['온도']]
inDep = lemon_pd[['판매량']]
print(dep.shape, inDep.shape)
 
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')
 
###########################
# Learning                                       # 
###########################
model.fit(dep, inDep, epochs=1000, verbose=0)
model.fit(dep, inDep, epochs=10)
 
###########################
# Prediction                                     # 
###########################
print(model.predict(dep))
print(model.predict([[15]]))