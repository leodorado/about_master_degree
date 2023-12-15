pip install tensorflow

pip install flax==0.7.5 jax==0.4.20 jaxlib==0.4.20+cuda11.cudnn86 numba==0.58.1 plotnine==0.12.4 scipy==1.11.3 tensorflow==2.14.0


!pip install numpy==1.26.2


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Bidirectional, LSTM

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Define the directory where your files are located
directory = 'no_epileptic_10'

# Initialize a list to store the selected data
selected_data = []

# Define the target number of observations (3600 in your case)
target_num_observations = 3600

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        data = np.loadtxt(file_path)

        # Check if the data has the desired number of observations
        num_observations = data.shape[0]

        if num_observations == target_num_observations:
            # Select the first 20 columns and add it to the selected_data list
            selected_data.append(data[:, :92])

# Convert the list of selected data into a NumPy array
input_data_3 = np.array(selected_data)
print(input_data_3.shape)

# Define the directory where your files are located

directory = 'epileptic_10'

# Initialize a list to store the selected data
selected_data = []

# Define the target number of observations (3600 in your case)
target_num_observations = 3600

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        data = np.loadtxt(file_path)

        # Check if the data has the desired number of observations
        num_observations = data.shape[0]

        if num_observations == target_num_observations:
            # Select the first 20 columns and add it to the selected_data list
            selected_data.append(data[:, :92])

# Convert the list of selected data into a NumPy array
input_data_4 = np.array(selected_data)
print(input_data_4.shape)


# Concatenate input_data and input_data_2 along the first axis (stack them)
X_2 = np.concatenate((input_data_3, input_data_4), axis=0)
X_2.shape


# Create the binary output array y
num_samples_input_data_3 = input_data_3.shape[0]
num_samples_input_data_4 = input_data_4.shape[0]
print(num_samples_input_data_3)
print(num_samples_input_data_4)


y_2 = np.concatenate((np.zeros(num_samples_input_data_3), np.ones(num_samples_input_data_4)))
y_2 = y_2.astype(int)
y_2.shape
#print(y_2)


# Assuming you have your features in X and binary labels in y
# Count the number of ones (positive class) in y
num_ones = np.sum(y_2 == 1)
print(num_ones)

# Find the indices of zeros (negative class) in y
zero_indices = np.where(y_2 == 0)[0]
print(zero_indices)

# Randomly sample the same number of zeros as ones
selected_zero_indices = np.random.choice(zero_indices, size=num_ones, replace=False)
print(selected_zero_indices)

# Combine the indices of ones and selected zeros
selected_indices = np.concatenate([np.where(y_2 == 1)[0], selected_zero_indices])
print(selected_indices)

# Create the subsample of X and y
X_subsample_2 = X_2[selected_indices]
X_subsample_2.shape
#print(X_subsample_2)
y_subsample_2 = y_2[selected_indices]
#print(y_subsample_2)


# Split X and y into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_subsample_2 , y_subsample_2, test_size=0.2, random_state=42)



# Reshape the data to 2D before scaling
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to your training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape the scaled data back to its original shape
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)


X_train_scaled.shape[2]


# Save X_train, y_train, X_test, and y_test
np.save('X_train_scaled.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test_scaled.npy', X_test)
np.save('y_test.npy', y_test)


# Load the saved arrays
X_train_scaled = np.load('X_train_scaled.npy')
print(type (X_train_scaled))
#print(X_train_reshaped.ndim)
print(len(X_train_scaled))
print(X_train_scaled.shape)
y_train = np.load('y_train.npy')
print(' ')
print(type (y_train))
print(y_train.ndim)
print(len(y_train))
print(y_train.shape)
print(y_train)
X_test_scaled = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

print(' ')
print("Ejemplos de datos de entrenamiento:")
print(' ')
print(X_train_scaled[0])
print("Etiqueta correspondiente:")
print(y_train[0])

# Para tus datos de entrenamiento
unique, counts = np.unique(y_train, return_counts=True)
class_counts_train = dict(zip(unique, counts))

# Para tus datos de prueba
unique, counts = np.unique(y_test, return_counts=True)
class_counts_test = dict(zip(unique, counts))

print("Clases y sus frecuencias en datos de entrenamiento:")
print(class_counts_train)

print("\nClases y sus frecuencias en datos de prueba:")
print(class_counts_test)


1# Define your LSTM model
model = keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(256, activation=keras.layers.LeakyReLU(alpha=0.05), return_sequences=True), input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(layers.LSTM(164, activation=keras.layers.LeakyReLU(alpha=0.05)))

model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=128, validation_data=(X_test_scaled, y_test))


# Make predictions on the test data
y_pred = model.predict(X_test_scaled)
print(y_pred)
# If you want to convert the probabilities to binary predictions (0 or 1), you can use a threshold, e.g., 0.5:
y_pred_binary = (y_pred > 0.47).astype(int)

print(y_pred_binary)


from sklearn.metrics import confusion_matrix, accuracy_score

# If you have true labels as y_test (ground truth) and y_pred_binary as your predicted binary labels
confusion = confusion_matrix(y_test, y_pred_binary)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)

# Calculate sensitivity (true positive rate)
sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])

# Calculate specificity (true negative rate)
specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])

print("Confusion Matrix:\n", confusion)
print("Accuracy:", accuracy)
print("Sensitivity (True Positive Rate):", sensitivity)
print("Specificity (True Negative Rate):", specificity)