from analisis import cargar_datos, preparar_variables, analisis_clusters, agregar_variable_activo
import pandas as pd
import matplotlib.pyplot as plt
# 1. Cargar datos
df = cargar_datos()

# 2. Preparar meses comprados
meses_por_cliente = preparar_variables(df)

# 3. Hacer clustering y obtener los DataFrames relevantes
df_clientes, productos_por_cluster, meses_cluster, k_optimo = analisis_clusters(df, meses_por_cliente)

# 4. Agregar la variable binaria 'Activo' e imprimir inactivos
df_modelo, total_activos, total_inactivos = agregar_variable_activo(df_clientes, meses_por_cliente)
#Cargar el archivo de clientes inactivos
df_inactivos = pd.read_csv("clientes_inactivos.csv")
total_inactivos = len(df_inactivos)

# Supongamos que hay 1500 clientes activos más
total_clientes = total_activos + total_inactivos

# Crear DataFrame para el gráfico de pastel
data_pie = pd.DataFrame({
    'Estado': ['Activos', 'Inactivos'],
    'Cantidad': [total_activos, total_inactivos]
})

# Graficar pastel
plt.figure(figsize=(6, 6))
plt.pie(data_pie['Cantidad'], labels=data_pie['Estado'], autopct='%1.1f%%', startangle=90)
plt.title('Distribución de Clientes Activos vs Inactivos')
plt.tight_layout()
plt.show()