# Importar librerías
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar el archivo CSV
csv_path = "./Data/1. Ventas 2022 ENE-MAR.csv"  # Parameterize the file path here
df = pd.read_csv(csv_path, encoding="latin1")

# Agrupar por cliente con métricas relevantes
df_clientes = df.groupby('ID Cliente').agg({
    'Venta USD': 'sum',
    'Venta Cajas': 'sum',
    'Producto': 'nunique',
    'Categoría': 'nunique',
    'Mes': 'nunique'
}).reset_index()

# Renombrar columnas para claridad
df_clientes.columns = ['ID Cliente', 'Total Ventas USD', 'Total Cajas',
                       'Productos Diferentes', 'Categorías Diferentes', 'Meses Diferentes']

# Normalizar las columnas numéricas
features = df_clientes.drop('ID Cliente', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Aplicar modelo de clustering KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
df_clientes['Cluster'] = kmeans.fit_predict(X)

# Crear resumen por cluster con promedios
cluster_summary = df_clientes.groupby('Cluster')[[
    'Total Ventas USD', 'Total Cajas', 'Productos Diferentes',
    'Categorías Diferentes', 'Meses Diferentes'
]].mean().round(2)

# Contar cuántos clientes hay por cluster
cluster_counts = df_clientes['Cluster'].value_counts().sort_index()

# Agregar conteo de clientes al resumen
cluster_summary['Número de Clientes'] = cluster_counts.reindex(cluster_summary.index, fill_value=0)

# Mostrar resumen
print("Resumen por Cluster:")
print(cluster_summary)

# Guardar resultados a CSV solo si se desea
guardar_csv = False  # Cambia a True si quieres guardar los archivos

if guardar_csv:
    df_clientes.to_csv("clientes_segmentados.csv", index=False)
    cluster_summary.to_csv("resumen_clusters.csv")
