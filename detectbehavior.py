# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Cargar el archivo CSV
csv_path = "./Data/Venta 2022 OCT - DIC V3.csv"
df = pd.read_csv(csv_path, encoding="latin1")

# Agrupar por cliente con métricas relevantes
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

# ---------- MÉTODO DEL CODO + KNEED ----------
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Encontrar el "codo" automáticamente
kl = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
k_optimo = kl.elbow
print(f"\nNúmero óptimo de clusters según Kneedle: {k_optimo}")

# Graficar curva del codo con el punto marcado
plt.plot(K_range, inertias, marker='o')
plt.vlines(k_optimo, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inercia (Within-Cluster SSE)")
plt.title("Método del Codo para Selección Automática de k")
plt.grid(True)
plt.show()

# ---------- CLUSTERING CON K ÓPTIMO ----------
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init="auto")
df_clientes['Cluster'] = kmeans.fit_predict(X)

# Resumen por cluster
cluster_summary = df_clientes.groupby('Cluster')[[
    'Total Ventas USD', 'Total Cajas', 'Productos Diferentes',
    'Categorías Diferentes', 'Meses Diferentes'
]].mean().round(2)

# Tamaño de cada cluster
cluster_counts = df_clientes['Cluster'].value_counts().sort_index()
cluster_summary['Número de Clientes'] = cluster_counts.reindex(cluster_summary.index, fill_value=0)

# Mostrar resumen
print("\nResumen por Cluster:")
print(cluster_summary)

# Guardar archivos opcionalmente
guardar_csv = True  # Cambiar a False si no se desea guardar
# csv_path = "./Data/1. Ventas 2023 ENE/"
if guardar_csv:
    cluster_summary.to_csv(csv_path + " resumen_clusters.csv")
