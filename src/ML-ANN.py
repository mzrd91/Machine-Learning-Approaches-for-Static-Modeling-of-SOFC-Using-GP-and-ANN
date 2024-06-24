# Artificial Neural Network

X_train, Y_train, X_test, Y_test = generate_data()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
train_mae_list, train_mse_list, test_mae_list, test_mse_list = [], [], [], []

model = keras.Sequential([
    keras.layers.Input(shape=(2,)), # Input layer with 2 features (temperature and current)
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)), # L2 regularization
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)), # L2 regularization
    keras.layers.Dense(1) # Output layer with a single neuron for regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), verbose=0)

train_loss, train_mae = model.evaluate(X_train, Y_train, verbose=0)
train_mae_list.append(train_mae)
train_mse_list.append(train_loss)

test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
test_mae_list.append(test_mae)
test_mse_list.append(test_loss)

Y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(Y_train, Y_train_pred)
train_mse_list.append(train_mse)

Y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(Y_test, Y_test_pred)
test_mse_list.append(test_mse)

weights = model.get_weights()

T = 900
current = 500

input_layer = np.array([T, current])

def get_value(T, current):
    # Use the weights and biases to construct the equation
    hidden_layer1 = np.dot(input_layer, weights[0]) + weights[1]
    hidden_layer1 = np.maximum(0, hidden_layer1) # ReLU activation
    hidden_layer2 = np.dot(hidden_layer1, weights[2]) + weights[3]
    hidden_layer2 = np.maximum(0, hidden_layer2) # ReLU activation
    output = np.dot(hidden_layer2, weights[4]) + weights[5]
    return output

def construct_equation():
    equation = f"output = ReLU({weights[0][0]} * T + {weights[1][0]}) # First hidden layer\n"
    equation += f"output = ReLU({weights[2][0]} * output + {weights[3][0]}) # Second hidden layer\n"
    equation += f"output = {weights[4][0]} * output + {weights[5][0]} # Output layer"
    return equation

print("MAE on Training Data \n\nBest: {}\nWorst: {}\nMean: {} \nVariance: {}\n".format(*calculate_statistics(train_mae_list)))
print("MSE on Training Data \n\nBest: {}\nWorst: {}\nMean: {} \nVariance: {}\n".format(*calculate_statistics(train_mse_list)))
print("MAE on Testing Data \n\nBest: {}\nWorst: {}\nMean: {} \nVariance: {}\n".format(*calculate_statistics(test_mae_list)))
print("MSE on Testing Data \n\nBest: {}\nWorst: {}\nMean: {} \nVariance: {}\n".format(*calculate_statistics(test_mse_list)))

print("\nBest Value (with respect to test data): {}".format(str(get_value(T, current)[0])))

equation = construct_equation()
print("\nBest Equation (with respect to test data): {}".format(construct_equation()))
