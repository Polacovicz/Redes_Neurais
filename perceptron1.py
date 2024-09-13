#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:19:25 2024

@author: julio
"""

# Importa a biblioteca numpy, que fornece suporte para arrays e operações matemáticas eficientes
import numpy as np

# Entradas da rede neural, onde cada sublista representa uma combinação de entrada
# São os possíveis valores de entrada do problema AND (com 2 entradas binárias)
entradas = np.array([[0,0], [0,1], [1,0], [1,1]])

# Saídas esperadas, que correspondem às saídas corretas da operação AND para as entradas
# No problema AND, o resultado só é 1 quando ambas as entradas são 1, caso contrário, é 0
saidas = np.array([0, 0, 0, 1])

# Inicializamos os pesos com valores 0.0 para cada uma das entradas
# Como estamos lidando com duas entradas, temos dois pesos
pesos = np.array([0.0, 0.0])

# Definimos a taxa de aprendizagem, que controla o quanto os pesos devem ser ajustados
# durante o treinamento. Aqui, ela é definida como 0.1
taxaAprendizagem = 0.1

# Função de ativação Step Function (Função Degrau)
# Ela decide se o neurônio deve ser ativado (1) ou não (0)
# A ativação só ocorre se a soma ponderada das entradas for maior ou igual a 1
def stepFunction(soma):
    if (soma >= 1):
        return 1  # Neurônio é ativado
    return 0  # Neurônio não é ativado

# Função que calcula a saída do neurônio para um registro (um conjunto de entradas)
# Aqui usamos a função dot product (produto escalar) para multiplicar os valores das entradas
# pelos seus respectivos pesos
def calculaSaida(registro):
    s = registro.dot(pesos)  # Calcula a soma ponderada das entradas pelos pesos
    return stepFunction(s)  # Aplica a função degrau à soma ponderada

# Função de treinamento do perceptron
# A ideia é ajustar os pesos do perceptron até que ele seja capaz de classificar todas
# as entradas corretamente (erro total = 0)
def treinar():
    # Inicializamos erroTotal com um valor maior que 0 para que o laço inicie
    erroTotal = 1
    
    # Enquanto ainda houver erros (erroTotal for diferente de 0)
    while (erroTotal != 0):
        erroTotal = 0  # Zera o erro total no início de cada iteração
        
        # Loop que percorre todas as entradas e suas respectivas saídas esperadas
        for i in range(len(saidas)):
            # Calcula a saída do perceptron para a i-ésima entrada
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            
            # Calcula o erro: a diferença entre a saída esperada e a saída calculada
            erro = abs(saidas[i] - saidaCalculada)
            
            # Acumula o erro total, que será usado para decidir se o treinamento continua
            erroTotal += erro
            
            # Atualiza os pesos com base no erro e nas entradas
            for j in range(len(pesos)):
                # Regra de atualização dos pesos: peso = peso + (taxa de aprendizagem * entrada * erro)
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                # Exibe o peso atualizado no console
                print('Peso atualizado: ' + str(pesos[j]))
            
            # Exibe o total de erros ao final de cada iteração
            print('Total de erros: ' + str(erroTotal))

# Inicia o processo de treinamento do perceptron
treinar()
