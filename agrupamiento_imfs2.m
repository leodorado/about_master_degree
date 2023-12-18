clc;
clear all;
close all;

% Especifica la carpeta que contiene los archivos
carpeta_archivos = 'C:\Users\USER\Desktop\Oversampling\Nueva carpeta'; % Cambia esto a la ruta de tu carpeta

% Lista de archivos reducidos de IMF y P
archivos_imf = dir(fullfile(carpeta_archivos, 'reduced_to_2_imfimf_P*_col*.txt'));

% Iterar sobre cada archivo
for i = 170:length(archivos_imf)
    % Construir la ruta completa al archivo actual
    ruta_archivo = fullfile(carpeta_archivos, archivos_imf(i).name);

    % Extraer la categoría del nombre del archivo
    partes_nombre = strsplit(archivos_imf(i).name, '_');
    % Buscar el índice que contiene 'K' en el nombre
    indice_K = find(contains(partes_nombre, 'P'), 1);
    
    if ~isempty(indice_K)
        % Obtener la categoría desde el índice encontrado
        categoria_actual = str2double(partes_nombre{indice_K}(2:end));
        
        % Verificar si la categoría es un número válido
        if ~isnan(categoria_actual)
            % Construir la ruta de los archivos de la categoría actual
            archivos_categoria = dir(fullfile(carpeta_archivos, ['reduced_to_2_imfimf_P' num2str(categoria_actual) '_col*.txt']));

            % Verificar si hay archivos en la categoría actual
            if ~isempty(archivos_categoria)
                % Inicializar una matriz para almacenar todas las columnas de la categoría
                todas_columnas_categoria = [];

                % Iterar sobre cada archivo de la categoría actual
                for j = 1:length(archivos_categoria)
                    % Construir la ruta completa al archivo actual
                    ruta_archivo_categoria = fullfile(carpeta_archivos, archivos_categoria(j).name);

                    % Leer el archivo actual y concatenar las columnas
                    columnas_actuales = dlmread(ruta_archivo_categoria);
                    todas_columnas_categoria = [todas_columnas_categoria, columnas_actuales];
                end

                % Construir el nombre del nuevo archivo combinado
                nombre_nuevo_archivo_categoria = ['red_2IMF_P' num2str(categoria_actual) '_comb.txt'];

                % Guardar todas las columnas combinadas en un solo archivo
                dlmwrite(nombre_nuevo_archivo_categoria, todas_columnas_categoria, 'delimiter', '\t', 'precision', 3);

                disp(['Proceso completado para combinar todas las columnas de reduced_to_2_imfimf_P' num2str(categoria_actual) '.']);
            else
                disp(['No se encontraron archivos de reduced_to_2_imfimf_P' num2str(categoria_actual) ' en la carpeta.']);
            end
        else
            disp(['El archivo ' archivos_imf(i).name ' no sigue el patrón esperado.']);
        end
    else
        disp(['No se encontró la letra "K" en el nombre del archivo ' archivos_imf(i).name]);
    end
end

disp('Proceso completo para todos los archivos en la carpeta.');

