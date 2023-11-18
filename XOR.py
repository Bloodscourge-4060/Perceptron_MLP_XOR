import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación
def derivada_sigmoide(x):
    return x * (1 - x)

# Lectura de los datos de entrenamiento y prueba desde archivos CSV
train_data = pd.read_csv('XOR_trn.csv')
test_data = pd.read_csv('XOR_tst.csv')

# Separar las columnas en características y etiquetas
X_train = train_data.iloc[:, :-1].values  # Características
y_train = train_data.iloc[:, -1].values  # Etiquetas
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Arquitectura de la red neuronal
input_size = X_train.shape[1]
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Inicialización de pesos y bias
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_input_hidden = np.random.uniform(size=(1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_hidden_output = np.random.uniform(size=(1, output_size))

# Entrenamiento
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_input_hidden
    hidden_layer_output = sigmoide(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
    output = sigmoide(output_layer_input)
    
    # Cálculo del error
    error = y_train.reshape(-1, 1) - output
    
    # Backpropagation
    d_output = error * derivada_sigmoide(output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * derivada_sigmoide(hidden_layer_output)
    
    # Actualización de pesos y bias
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_hidden_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_input_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Predicción en datos de prueba
hidden_layer_test = sigmoide(np.dot(X_test, weights_input_hidden) + bias_input_hidden)
predicted_output = sigmoide(np.dot(hidden_layer_test, weights_hidden_output) + bias_hidden_output)

# Redondear las predicciones para el problema binario
predicted_output = np.round(predicted_output)

mse = np.mean((y_test.reshape(-1, 1) - predicted_output) ** 2)
print(f"MSE: {mse}")

plt.scatter(X_test[:,0], X_test[:,1], c=predicted_output.ravel(), cmap='viridis')
plt.title('Predicciones')
plt.show()
