#Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import glob
import os

# Ruta a la carpeta donde están todos los archivos CSV
ruta_carpeta = "./Data/"

# Buscar todos los archivos CSV en la carpeta
archivos = glob.glob(os.path.join(ruta_carpeta, "*.csv"))

# Leer y concatenar todos los archivos, agregando el nombre del archivo como columna "Origen"
lista_dfs = []
for archivo in archivos:
    temp_df = pd.read_csv(archivo, encoding="latin1")
    temp_df["Origen"] = os.path.basename(archivo)  # Agrega columna con el nombre del archivo
    lista_dfs.append(temp_df)

df = pd.concat(lista_dfs, ignore_index=True)



# Ver los meses en los que compró cada cliente
meses_por_cliente = df.groupby('ID Cliente')['Mes'].unique().reset_index()
meses_por_cliente.columns = ['ID Cliente', 'Meses Comprados']

# Obtener tamaño de cliente único por ID Cliente
tamano_por_cliente = df[['ID Cliente', 'Tamaño de Cliente']].drop_duplicates()

# Unir con meses_por_cliente
meses_por_cliente = meses_por_cliente.merge(tamano_por_cliente, on='ID Cliente', how='left')

# Reordenar columnas si deseas
meses_por_cliente = meses_por_cliente[['ID Cliente', 'Tamaño de Cliente', 'Meses Comprados']]

# Guardar archivo final
meses_por_cliente.to_csv("meses_por_cliente.csv", index=False)

# Ordenar los meses alfabéticamente si es necesario
meses_por_cliente['Meses Comprados'] = meses_por_cliente['Meses Comprados'].apply(lambda x: sorted(x))

meses_por_cliente['N° Meses Comprados'] = meses_por_cliente['Meses Comprados'].apply(len)
# Guardar el DataFrame de meses por cliente a un archivo CSV
meses_por_cliente.to_csv("meses_por_cliente.csv", index=False)

# ========== ANÁLISIS PRELIMINAR ==========

# Analisis preliminar de clientes
conteo_tamano = df.groupby("Tamaño de Cliente")["ID Cliente"].nunique().reset_index()
conteo_tamano.columns = ["Tamaño de Cliente", "Número de Clientes"]

print(conteo_tamano)

# Confirmar que todo se cargó correctamente
print(f"{len(archivos)} archivos combinados. Total de registros: {len(df)}")
# for archivo in archivos:
#     print(df.head())

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
                       'Productos Diferentes', 'Categorías Diferentes', 'Meses Comprados']

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

# ----------Asignar los clusters al DataFrame original-----------
# MERGE para saber a qué cluster pertenece cada fila de compra original
productos_por_cluster = df.merge(df_clientes[['ID Cliente', 'Cluster']], on='ID Cliente')
# Unir la información de meses comprados con los clusters
meses_cluster = meses_por_cliente.merge(df_clientes[['ID Cliente', 'Cluster']], on='ID Cliente', how='left')
# Promedio de meses comprados por cluster
promedio_meses_cluster = meses_cluster.groupby('Cluster')['N° Meses Comprados'].mean().round(2).reset_index()
print("\nPromedio de meses en los que compran por cluster:")
print(promedio_meses_cluster)
# Agrupar por cluster y calcular métricas
tamano_cluster = meses_cluster.groupby(['Cluster', 'Tamaño de Cliente'])['ID Cliente'].count().unstack(fill_value=0)
print("\nDistribución de tamaños de cliente por cluster:")
print(tamano_cluster)

# Expandir los meses a filas individuales
meses_explotado = meses_cluster.explode('Meses Comprados')

# Contar meses por cluster
conteo_meses = meses_explotado.groupby(['Cluster', 'Meses Comprados'])['ID Cliente'].count().reset_index()
conteo_meses.columns = ['Cluster', 'Mes', 'Clientes']

# ----Graficas de distribución de clientes por meses dependiendo del cluster----
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.barplot(data=conteo_meses, x='Mes', y='Clientes', hue='Cluster')
plt.title("Distribución de compras por mes y clientes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# TOP productos por cluster en términos de cajas vendidas
top_productos = productos_por_cluster.groupby(['Cluster', 'Producto'])['Venta Cajas'].sum().reset_index()
top_productos = top_productos.sort_values(['Cluster', 'Venta Cajas'], ascending=[True, False])

# Mostrar los 5 productos más vendidos por cluster como ejemplo
for cl in sorted(top_productos['Cluster'].unique()):
    print(f"\nTop productos del Cluster {cl}:")
    print(top_productos[top_productos['Cluster'] == cl].head(5))

# Resumen por cluster
cluster_summary = df_clientes.groupby('Cluster')[[
    'Total Ventas USD', 'Total Cajas', 'Productos Diferentes',
    'Categorías Diferentes', 'Meses Comprados'
]].mean().round(2)

# Tamaño de cada cluster
cluster_counts = df_clientes['Cluster'].value_counts().sort_index()
cluster_summary['Número de Clientes'] = cluster_counts.reindex(cluster_summary.index, fill_value=0)

# Mostrar resumen
print("\nResumen por Cluster:")
print(cluster_summary)

# Guardar archivos opcionalmente
guardar_csv = True  # Cambiar a False si no se desea guardar
if guardar_csv:
    cluster_summary.to_csv("resumen_clustersTotal.csv")






