import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import glob
import os

def cargar_datos(ruta_carpeta="./Data/"):
    archivos = glob.glob(os.path.join(ruta_carpeta, "*.csv"))
    lista_dfs = []
    for archivo in archivos:
        temp_df = pd.read_csv(archivo, encoding="latin1")
        temp_df["Origen"] = os.path.basename(archivo)
        lista_dfs.append(temp_df)
    df = pd.concat(lista_dfs, ignore_index=True)
    return df

def preparar_variables(df):
    meses_por_cliente = df.groupby('ID Cliente')['Mes'].unique().reset_index()
    meses_por_cliente.columns = ['ID Cliente', 'Meses Comprados']
    meses_por_cliente['Meses Comprados'] = meses_por_cliente['Meses Comprados'].apply(lambda x: sorted(x))
    meses_por_cliente['N° Meses Comprados'] = meses_por_cliente['Meses Comprados'].apply(len)

    tamano_por_cliente = df[['ID Cliente', 'Tamaño de Cliente']].drop_duplicates()
    meses_por_cliente = meses_por_cliente.merge(tamano_por_cliente, on='ID Cliente', how='left')
    meses_por_cliente = meses_por_cliente[['ID Cliente', 'Tamaño de Cliente', 'Meses Comprados', 'N° Meses Comprados']]
    meses_por_cliente.to_csv("meses_por_cliente.csv", index=False)
    return meses_por_cliente

def analisis_clusters(df, meses_por_cliente):
    df_clientes = df.groupby('ID Cliente').agg({
        'Venta USD': 'sum',
        'Venta Cajas': 'sum',
        'Producto': 'nunique',
        'Categoría': 'nunique',
        'Mes': 'nunique'
    }).reset_index()

    df_clientes.columns = ['ID Cliente', 'Total Ventas USD', 'Total Cajas',
                           'Productos Diferentes', 'Categorías Diferentes', 'Meses Comprados']

    features = df_clientes.drop('ID Cliente', axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
    k_optimo = kl.elbow

    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init="auto")
    df_clientes['Cluster'] = kmeans.fit_predict(X)

    productos_por_cluster = df.merge(df_clientes[['ID Cliente', 'Cluster']], on='ID Cliente')
    meses_cluster = meses_por_cliente.merge(df_clientes[['ID Cliente', 'Cluster']], on='ID Cliente', how='left')

    return df_clientes, productos_por_cluster, meses_cluster, k_optimo

def agregar_variable_activo(df_clientes, meses_por_cliente):
    def cliente_activo(meses):
        return int(meses > 6)

    meses_por_cliente['Activo'] = meses_por_cliente['N° Meses Comprados'].apply(cliente_activo)
    inactivos = meses_por_cliente[meses_por_cliente['Activo'] == 0]
    activos = meses_por_cliente[meses_por_cliente['Activo'] == 1]

    print("Clientes inactivos (compraron en 6 meses o menos):")
    print(inactivos[['ID Cliente', 'Tamaño de Cliente', 'N° Meses Comprados']])
    print(f"\nTotal de clientes inactivos: {len(inactivos)}")

    inactivos.to_csv("clientes_inactivos.csv", index=False)

    df_modelo = df_clientes.merge(meses_por_cliente[['ID Cliente', 'Activo']], on='ID Cliente')

    # Crear dataset con variables predictoras
    df_modelo[['ID Cliente', 'Total Ventas USD', 'Total Cajas',
           'Productos Diferentes', 'Categorías Diferentes', 'Activo']].to_csv("clientes_modelo.csv", index=False)

    return df_modelo, len(activos), len(inactivos)