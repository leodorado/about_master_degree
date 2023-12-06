% Este codigo permite cargar la base de datos original obtenida 
% desde Physionet
% Se encontrará en el Workspace tanto las señales sin convulsiones, como
% las que tienen convulsiones.
close all;
clear all;
clc;
% Señal EEG durante tarea de calculo matemático
load('EEG_Classification_Final.mat')
signal  =  S_EEG_F;

