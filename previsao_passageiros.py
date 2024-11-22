import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# Dados de passageiros (baseado no conjunto de dados "Airline Passengers")
# Número de passageiros de uma companhia aérea entre 1949 e 1960
data = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 142, 131, 125,
    149, 170, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 178, 199, 199, 199, 198, 181, 183, 187, 191, 194, 196
]

# Normalizando os dados para o intervalo [0, 1] usando MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data = np.array(data).reshape(-1, 1)
data_scaled = scaler.fit_transform(data)

# Criar os conjuntos de dados de treinamento
# Vamos usar 12 meses de dados anteriores para prever o mês seguinte
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12  # Usaremos 12 meses de dados para prever o próximo mês
X, y = create_dataset(data_scaled, time_step)

# Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Remodelar os dados para [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Criar o modelo de rede neural LSTM (Long Short-Term Memory)
model = models.Sequential([
    layers.LSTM(64, return_sequences=False, input_shape=(time_step, 1)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Avaliar o modelo
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Fazer previsões com o modelo
predictions = model.predict(X_test)

# Inverter a normalização dos dados para obter valores reais
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotar as previsões vs valores reais
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Real Data')
plt.plot(predictions_rescaled, label='Predicted Data')
plt.title('Airline Passengers Prediction')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

    
    