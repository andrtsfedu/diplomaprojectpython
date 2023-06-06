import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Генерация синтетических данных
X, y = make_blobs(n_samples=200, centers=4, random_state=0, cluster_std=0.7)

# Инициализация и обучение модели k-средних
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Получение предсказанных меток кластеров
predicted_labels = kmeans.predict(X)

# Визуализация результатов
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
plt.title("k-средних")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()