import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#Implementação do algoritmo de K-vizinhos próximos
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    #Cálculo da distância euclidiana de dois vetores
    def euclidean_distance(self, x, val):
        return (sum([(p-q)**2 for p,q in zip(x,val)]))**0.5

    #Salva os dados no modelo
    def fit(self, X, y):
        self.X = X
        self.y = y.tolist()

    #Encontra os K vizinhos mais próximmos de cada vetor em value
    def predict(self, value):
        result = []
        for val in value:
            predicted_labels = []
            predicted_distances = []

            for j in range(len(self.X)):
                distance = self.euclidean_distance(self.X.iloc[j].tolist(), val)

                if len(predicted_distances) < self.k:
                    predicted_distances.append(distance)
                    predicted_labels.append(self.y[j])

                else:
                    if max(predicted_distances) > distance:
                        maxIndex = predicted_distances.index(max(predicted_distances))
                        predicted_distances[maxIndex] = distance
                        predicted_labels[maxIndex] = self.y[j]

            #Ordena os k mais próximos de forma ascendente
            sorted_labels = sorted([(dist,label) for label,dist in zip(predicted_labels, predicted_distances)])

            #Conta quantas vezes cada rótulo aparece nos pontos mais próximos
            counter = dict()
            for label in predicted_labels:
                if label in counter.keys():
                    counter[label] += 1
                else:
                    counter[label] = 1

            # Ordena os rótulos de forma ascendente pela frequência
            counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}

            maxLabels = []

            #Salva os maiores rótulos em maxLabels pela frequência
            for i in counter.keys():
                if counter[i] == max(counter.values()):
                    maxLabels.append(i)
                else:
                    break

            #Se só houver 1 rótulo, retorne-o. Caso houver mais de 1, encontre qual está mais próximo do ponto a ser classificado
            if len(maxLabels) == 1:
                result.append(maxLabels[0])
            else:
                for item in sorted_labels:
                    if item[1] in maxLabels:
                        result.append(item[1])
                        break

        return result
                
#Importando dados
data = pd.read_csv("winequality-red.csv", sep=';')

#Separa os rótulos dos dados
X = data.drop(columns=['quality'])
y = data['quality']

#Inicializando o modelo feito acima e o modelo da biblioteca scikit-learn para teste
knn = KNearestNeighbors(5)
knnsk = KNeighborsClassifier(5)

#"Treinamento" do modelo
knn.fit(X, y)
knnsk.fit(X, y)

#Dados para teste
test_data = [
    [4.3, 0.12, 0.05, 12.5, 0.130, 3.1, 45, 0.998, 3.8, 1.2, 9.7],
    [0.3, 1.15, 0.22, 2.9, 0.020, 2.8, 19, 0.978, 1.1, 5.6, 1.9],
    [5.7, 0.33, 0.88, 15.0, 0.077, 0.9, 50, 0.995, 4.5, 0.8, 11.2],
    [1.4, 0.68, 0.11, 6.3, 0.056, 1.5, 27, 0.984, 2.2, 4.0, 5.4],
    [2.0, 0.90, 0.30, 7.7, 0.095, 1.2, 36, 0.999, 2.9, 3.3, 8.6],
    [0.7, 0.47, 0.14, 3.8, 0.045, 2.4, 21, 0.992, 1.6, 2.1, 2.0],
    [3.6, 0.55, 0.76, 10.2, 0.110, 3.5, 34, 0.996, 5.1, 1.9, 7.4],
    [1.1, 0.22, 0.19, 5.1, 0.033, 2.7, 23, 0.980, 2.0, 3.8, 4.8],
    [6.0, 0.39, 0.28, 9.5, 0.084, 1.8, 39, 0.994, 3.5, 2.4, 6.9],
    [2.9, 0.80, 0.52, 8.1, 0.072, 1.4, 30, 0.987, 4.2, 2.6, 10.0]
]

#Rodando as predições do modelo
print(knn.predict(test_data)) #Resultado = [7, 5, 6, 5, 5, 5, 5, 6, 5]
print(knnsk.predict(test_data)) #Resultado = [6, 5, 6, 5, 5, 5, 5, 6, 5]

#OBS: Resultados podem variar já que o critério de desempate dos algoritmos não é o mesmo nos dois algoritmos, como visto no primeiro item.