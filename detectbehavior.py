# Importar librerías
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Cargar archivo
csv_path = "./Data/1. Ventas 2022 ENE-MAR.csv"
df = pd.read_csv(csv_path, encoding="latin1")

# Agrupar por cliente
df_clientes = df.groupby('ID Cliente').agg({
    'Venta USD': 'sum',
    'Venta Cajas': 'sum',
    'Producto': 'nunique',
    'Categoría': 'nunique',
    'Mes': 'nunique'
}).reset_index()

# Renombrar columnas
df_clientes.columns = ['ID Cliente', 'Total Ventas USD', 'Total Cajas',
                       'Productos Diferentes', 'Categorías Diferentes', 'Meses Diferentes']

# Normalizar
features = df_clientes.drop('ID Cliente', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Evaluar Silhouette Score para distintos k
best_k = 2
best_score = -1
scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    if score > best_score:
        best_score = score
        best_k = k

# Mostrar gráfica de scores
plt.plot(K_range, scores, marker='o')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Selección automática de k con Silhouette")
plt.grid(True)
plt.show()

# Usar el mejor k encontrado
print(f"\nMejor número de clusters según Silhouette Score: {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df_clientes['Cluster'] = kmeans.fit_predict(X)

# Resumen por cluster
cluster_summary = df_clientes.groupby('Cluster')[[
    'Total Ventas USD', 'Total Cajas', 'Productos Diferentes',
    'Categorías Diferentes', 'Meses Diferentes'
]].mean().round(2)

# Tamaño de cada cluster
cluster_summary['Número de Clientes'] = df_clientes['Cluster'].value_counts().sort_index()

# Mostrar resumen
print("\nResumen por Cluster:")
print(cluster_summary)