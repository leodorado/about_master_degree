close all;
clear all;
clc;
% En este codigo se realiza el procedimiento EMD para las señales teniendo
% en cuenta que ahora se aplicará para todas las columnas de los archivos,
% ya que en el anterior codigo llamado emd_dataset se esta calculando la
% emd para el archivo como tal y no para cada una de las columnas que
% representan los canales del test EEG.
k_values = 1:357;

for i = k_values
    % Cargar el archivo correspondiente a K
    load(['K' num2str(i) '.txt']);

    a_K = readtable(['K' num2str(i) '.txt']);
    c_K = table2cell(a_K);
    fs = 1000;
    ts = 1/fs;
    tm = 1/10e2;

    % Iterar sobre cada columna
    for col = 1:size(c_K, 2)
        s_K = cell2mat(c_K(:, col));

        % Resto del código para EMD en s_K
        [imf_K, ~, ~] = emd(s_K);
        [m_K, n_K] = size(imf_K);
        
        % Ajustar el número de filas y columnas según el número de IMFs
        num_rows = ceil(sqrt(n_K));
        num_cols = ceil(n_K / num_rows);
        
        % Crear un archivo para guardar todas las IMFs de K
        fid_imf_K = fopen(['imf_K' num2str(i) '_col' num2str(col) '.txt'], 'w');
        
        % Guardar todas las IMFs en un solo archivo de texto para K
        for t = 1:length(s_K)
            % Escribir las columnas de imf_K en el archivo
            for j = 1:n_K
                fprintf(fid_imf_K, '%f\t', imf_K(t, j));
            end
            fprintf(fid_imf_K, '\n');
        end

        % Cerrar archivo
        fclose(fid_imf_K);
    end
    
    % Repetir el mismo proceso para el conjunto "P" si es necesario
    load(['P' num2str(i) '.txt']);

    a_P = readtable(['P' num2str(i) '.txt']);
    c_P = table2cell(a_P);

    % Iterar sobre cada columna para "P"
    for col = 1:size(c_P, 2)
        s_P = cell2mat(c_P(:, col));

        % Resto del código para EMD en s_P
        [imf_P, ~, ~] = emd(s_P);
        [m_P, n_P] = size(imf_P);
        
        % Ajustar el número de filas y columnas según el número de IMFs
        num_rows = ceil(sqrt(n_P));
        num_cols = ceil(n_P / num_rows);
        
        % Crear un archivo para guardar todas las IMFs de P
        fid_imf_P = fopen(['imf_P' num2str(i) '_col' num2str(col) '.txt'], 'w');
        
        % Guardar todas las IMFs en un solo archivo de texto para P
        for t = 1:length(s_P)
            % Escribir las columnas de imf_P en el archivo
            for j = 1:n_P
                fprintf(fid_imf_P, '%f\t', imf_P(t, j));
            end
            fprintf(fid_imf_P, '\n');
        end

        % Cerrar archivo
        fclose(fid_imf_P);
    end
end
