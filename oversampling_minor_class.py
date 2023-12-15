#1
''' En este codigo se realiza un sobremuestreo a la clase minoritaria que en este
 caso es la clase epiléptica ya que cuenta con 80 muestras mientras que la
clase no epiléptica tiene 357. Con la funcion SMOTE se logra el
propósito y se equilibran las clases.
'''
import os
import numpy as np

# Define the directory where your files are located
directory = 'no_epileptic'

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
print(input_data_3.shape)


# Define the directory where your files are located

directory = 'epileptic2'

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
print(input_data_4.shape)

#3
# Concatenate input_data and input_data_2 along the first axis (stack them)
X_2 = np.concatenate((input_data_3, input_data_4), axis=0)
X_2.shape


#4
# Create the binary output array y
# Distribucion de clases
num_samples_input_data_3 = input_data_3.shape[0]
num_samples_input_data_4 = input_data_4.shape[0]
print(num_samples_input_data_3)
print(num_samples_input_data_4)

#5
# Creacion de matriz de 1´s y 0´s
y_2 = np.concatenate((np.zeros(num_samples_input_data_3), np.ones(num_samples_input_data_4)))
y_2 = y_2.astype(int)
y_2.shape


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Reshape de X para que sea 2D
X_resampled = np.reshape(X_2, (437, -1))  # Ajusta el número de individuos en X
print('X_resampled antes de SMOTE: ', len(X_resampled))
# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_resampled, y_2)
print(' ')
print('X_resampled despues de SMOTE: ', len(X_resampled))
print('y_resampled despues de SMOTE: ', len(y_resampled))
# Reshape de nuevo a la forma original
X_resampled_reshaped = X_resampled.reshape(-1, 3600, 20)
print('X_resampled_reshaped: ', len(X_resampled_reshaped))
# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_reshaped, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)
print(' ')
print('Longitud de X_train: ', len(X_train))
print('Longitud de y_train: ', len(y_train))
print('Longitud de X_test: ', len(X_test))
print('Longitud de y_test: ', len(y_test))
print(len(X_resampled_reshaped[y_resampled == 1]))

# Definir el prefijo del nombre de archivo para la clase minoritaria sintética
prefix = 'P'

# Guardar los nuevos datos generados por SMOTE con nombres secuenciales
for i, data_sample in enumerate(X_resampled_reshaped[y_resampled == 1]):
    filename = f'{prefix}{i + 81}.txt'
    np.savetxt(filename, data_sample.reshape(-1, 20))



import os
import shutil
import zipfile

# Definir el número de muestras sintéticas
num_sinteticas = 277

# Supongamos que los archivos están en el directorio actual y tienen la extensión .txt
files_to_zip = [f'P{i}.txt' for i in range(81, 81 + num_sinteticas)]

# Crear un directorio temporal para almacenar los archivos antes de comprimir
temp_dir = 'temp_files'
os.makedirs(temp_dir, exist_ok=True)

# Copiar los archivos al directorio temporal
for file in files_to_zip:
    shutil.copy(file, os.path.join(temp_dir, file))

# Crear un archivo ZIP
zip_filename = 'muestras_sinteticas.zip'
with zipfile.ZipFile(zip_filename, 'w') as zip_file:
    for file in files_to_zip:
        zip_file.write(os.path.join(temp_dir, file), arcname=file)

# Eliminar el directorio temporal después de la compresión
shutil.rmtree(temp_dir)

# Mover el archivo ZIP al directorio actual (opcional)
shutil.move(zip_filename, os.path.join(os.getcwd(), zip_filename))

print(f'Archivo ZIP creado: {zip_filename}')

