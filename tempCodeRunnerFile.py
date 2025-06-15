df_clientes = df.groupby('ID Cliente').agg({
    'Venta USD': 'sum',
    'Venta Cajas': 'sum',
    'Producto': 'nunique',
    'Categoría': 'nunique',
    'Mes': 'nunique'
}).reset_index()