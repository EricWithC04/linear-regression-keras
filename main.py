import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

datos = pd.read_csv('altura_peso.csv', sep=',', header=0)

x = datos["Altura"].values
y = datos["Peso"].values

np.random.seed(2)
modelo = Sequential()

input_dim = 1
output_dim = 1

modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

sgd = SGD(learning_rate=0.00001)
modelo.compile(loss='mse', optimizer=sgd)

modelo.summary()

num_epochs = 100
batch_size = len(x)
historia = modelo.fit(x=x, y=y, epochs=num_epochs, batch_size=batch_size, verbose=1)

capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

plt.subplot(1,2,1)
plt.plot(historia.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')

y_regr = modelo.predict(x)
print('ECM = {:.2f}'.format(np.mean((y_regr-y)**2)))
plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr, color='red', label='linea de predicción')
plt.title('Datos originales y regresión lineal')
plt.show()

x_pred = np.array([175])
y_pred = modelo.predict(x_pred)
print(f"Peso predicho: {y_pred[0][0]:.2f} kg")