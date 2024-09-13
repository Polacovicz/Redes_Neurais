#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:19:25 2024

@author: julio
"""
# Este é o cabeçalho do script que define o interpretador Python e a codificação de caracteres como UTF-8.
# O bloco de comentário informa a data de criação e o autor do código.

import numpy as np
# Importa a biblioteca NumPy, usada para operações com arrays e cálculos matemáticos.

# Definindo o conjunto de entradas (valores de uma tabela verdade) para uma porta AND.
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Saídas esperadas correspondentes às entradas (comportamento de uma porta AND).
saidas = np.array([0, 0, 0, 1])

# Inicializa os pesos com valores zero, um para cada entrada.
pesos = np.array([0.0, 0.0])

# Define a taxa de aprendizagem, que controla a magnitude dos ajustes dos pesos.
taxaAprendizagem = 0.1

# Função de ativação (Step Function) que determina a saída da rede.
def stepFunction(soma):
    # Se a soma das entradas ponderadas for maior ou igual a 1, retorna 1.
    if (soma >= 1):
        return 1
    # Caso contrário, retorna 0.
    return 0

# Função que calcula a saída da rede para um conjunto de entradas.
def calculaSaida(registro):
    # Calcula o produto escalar entre as entradas (registro) e os pesos.
    s = registro.dot(pesos)
    # Aplica a função de ativação na soma calculada.
    return stepFunction(s)

# Função que treina o perceptron ajustando os pesos.
def treinar():
    # Inicializa o erro total com 1 para garantir que o loop comece.
    erroTotal = 1
    # Loop até que o erro total seja 0, ou seja, até que o perceptron esteja completamente treinado.
    while (erroTotal != 0):
        erroTotal = 0
        # Itera sobre cada entrada e a saída correspondente.
        for i in range(len(saidas)):
            # Calcula a saída da rede para a entrada atual.
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            # Calcula o erro como a diferença absoluta entre a saída esperada e a saída calculada.
            erro = abs(saidas[i] - saidaCalculada)
            # Acumula o erro total para verificar se a rede ainda precisa de ajustes.
            erroTotal += erro
            # Atualiza os pesos, um para cada entrada, com base no erro.
            for j in range(len(pesos)):
                # A regra de atualização de pesos: novo peso = peso atual + (taxa de aprendizagem * entrada * erro)
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                # Imprime o valor do peso atualizado.
                print('Peso atualizado: ' + str(pesos[j]))
            # Imprime o erro total acumulado após a iteração sobre todas as entradas.
        print('Total de erros: ' + str(erroTotal))

# Chama a função de treinamento, que ajusta os pesos da rede.
treinar()

# Após o treinamento, imprime que a rede neural foi treinada.
print('Rede neural treinada')

# Testa a rede neural com as 4 entradas possíveis e imprime as saídas calculadas.
print(calculaSaida(entradas[0]))  # Testa a entrada [0, 0]
print(calculaSaida(entradas[1]))  # Testa a entrada [0, 1]
print(calculaSaida(entradas[2]))  # Testa a entrada [1, 0]
print(calculaSaida(entradas[3]))  # Testa a entrada [1, 1]
