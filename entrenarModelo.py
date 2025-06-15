import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar el dataset preparado
df = pd.read_csv("clientes_modelo.csv")

# Separar variables predictoras y la variable objetivo
X = df[['Total Ventas USD', 'Total Cajas', 'Productos Diferentes', 'Categorías Diferentes']]
y = df['Activo']

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir y entrenar modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = modelo.predict(X_test)
print("Reporte de clasificación:\n")
print(classification_report(y_test, y_pred))