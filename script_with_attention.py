import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

#1

import os
import numpy as np

# Define the directory where your files are located
#directory = 'no_epiliptic_txts'
directory = 'database_balanced'

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
            selected_data.append(data[:, :20])

# Convert the list of selected data into a NumPy array
input_data_3 = np.array(selected_data)

#2
# Define the directory where your files are located
#directory = 'epiliptic_txts'
directory = 'database_balanced'

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
            selected_data.append(data[:, :20])

# Convert the list of selected data into a NumPy array
input_data_4 = np.array(selected_data)

#3
# Concatenate input_data and input_data_2 along the first axis (stack them)
X_2 = np.concatenate((input_data_3, input_data_4), axis=0)
X_2.shape

#4
# Create the binary output array y
num_samples_input_data_3 = input_data_3.shape[0]
num_samples_input_data_4 = input_data_4.shape[0]

#5
y_2 = np.concatenate((np.zeros(num_samples_input_data_3), np.ones(num_samples_input_data_4)))
y_2 = y_2.astype(int)
y_2.shape

#6
# Assuming you have your features in X and binary labels in y
# Count the number of ones (positive class) in y
num_ones = np.sum(y_2 == 1)

# Find the indices of zeros (negative class) in y
zero_indices = np.where(y_2 == 0)[0]

# Randomly sample the same number of zeros as ones
selected_zero_indices = np.random.choice(zero_indices, size=num_ones, replace=False)

# Combine the indices of ones and selected zeros
selected_indices = np.concatenate([np.where(y_2 == 1)[0], selected_zero_indices])

# Create the subsample of X and y
X_subsample_2 = X_2[selected_indices]
y_subsample_2 = y_2[selected_indices]

#7

# Split X and y into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_subsample_2 , y_subsample_2, test_size=0.2, random_state=42)

#8
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

#9
X_train_scaled.shape[2]

#10
# Save X_train, y_train, X_test, and y_test
np.save('X_train_scaled.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test_scaled.npy', X_test)
np.save('y_test.npy', y_test)

#11
# Load the saved arrays
X_train_scaled = np.load('X_train_scaled.npy')
y_train = np.load('y_train.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

#12

# Define input layer
inputs = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))


# Define LSTM layer with return_sequences=True
lstm_output = layers.LSTM(16, activation=keras.layers.LeakyReLU(alpha=0.05), return_sequences=True)(inputs)

# Apply Attention layer to the LSTM output
attention_output = Attention(use_scale=True)([lstm_output, lstm_output])

# Flatten the attention output
flatten_output = Flatten()(attention_output)

# Dense layer for classification
output = Dense(1, activation='sigmoid')(flatten_output)

# Create the model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=16, validation_data=(X_test_scaled, y_test))

#13
# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# If you want to convert the probabilities to binary predictions (0 or 1), you can use a threshold, e.g., 0.5:
y_pred_binary = (y_pred > 0.5).astype(int)

#14
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



print('hola leandro')

