import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Função responsável por calcular o valor de b1 na equação da reta b0 + b1*x.
def calculate_b1(X, labels):
    x = sum([(labels[i] - labels.mean())*(X[i] - X.mean()) for i in range(len(X))])
    y = sum([(X[i] - X.mean())**2 for i in range(len(X))])
    return x/y

#Função responsável por calcular o valor de b0 na equação da reta b0 + b1*x.
def calculate_b0(X, labels):
    return labels.mean() - calculate_b1(X, labels) * X.mean()

#Implementação de modelo de regressão linear simples.
class SimpleLinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    #Calcula os valores de b0 e b1 para definir a equação da reta.
    def fit(self, X, y):
        self.b1 = calculate_b1(X, y)
        self.b0 = calculate_b0(X, y)

    #Encontra o resultado de n na equação da reta definida pelos dados inseridos no treinamento.
    def predict(self, val):
        return self.b0 + self.b1*val
    
#Dados criados aleatoriamente. Seguem uma tendência linear crescente.
X = np.array([random.randint(i, i+6) for i in range(1, 100)])
y = np.array([random.randint(i+2, (i+25)) for i in range(1, 100)])

#Treinamento modelo regressão linear feito acima.
model = SimpleLinearRegression()
model.fit(X, y)

#Treinamento modelo regressão linear da biblioteca scikit-learn.
modelsk = LinearRegression()
modelsk.fit(X.reshape(-1, 1),y)

#Pontos usados no treinamento.
plt.scatter(X, y, color='grey')

#Linha de predição do modelo linear feito acima.
plt.plot(X, [model.predict(i) for i in X])

#Adicionado 0.5 para não sobrepor as linhas no gráfico.
#Linha de predição do modelo linear da biblioteca scikit-learn
plt.plot(X+0.5, [modelsk.predict(np.array([i]).reshape(-1, 1)) for i in X], color='#f59920')
#OBS: Tanto o modelo feito acima quanto o da biblioteca importada deram os MESMOS resultados. +0.5 para facilitar a vizualização da reta no gráfico.

plt.show()