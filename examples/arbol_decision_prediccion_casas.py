"""
Ejemplo completo_ decision tree (árbol de decisión)
Dataset: Predicción de precios de Casas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

#Configurar numpy para reproducibilidad
np.random.seed(42)
print("DATASET")

datos = {
    'tamano_m2': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                  150, 160, 170, 180, 190, 200, 210, 220, 230, 240],

    'num_habitaciones': [1, 1, 2, 2, 2, 3, 3, 3, 3, 4,
                         4, 4, 4, 5, 5, 5, 5, 6, 6, 6],

    'antiguedad_anos': [30, 25, 20, 15, 10, 25, 20, 15, 10, 5,
                        30, 25, 20, 15, 10, 5, 20, 15, 10, 5],

    'precio_miles': [150, 180, 200, 230, 260, 250, 280, 310, 340, 350,
                     320, 350, 380, 400, 430, 460, 420, 450, 480, 510]
}

df = pd.DataFrame(datos)

print("\nDataset completo (20 casas):")
print(df)

print("\nEstadisticas básicas:")
print(df.describe())

# Visualizar relaciones

fig,axes = plt.subplots(1,3, figsize=(18,5))

axes[0].scatter(df['tamano_m2'],df['precio_miles'],color='blue',s=100, alpha=0.6)
axes[0].set_xlabel('Tamaño (m2)', fontsize=12)
axes[0].set_ylabel('Precio (miles $)', fontsize = 12)
axes[0].set_title('Tamaño vs Precio', fontweight='bold')
axes[0].grid(True, alpha =0.3)

axes[1].scatter(df['num_habitaciones'], df['precio_miles'],color='green', s=100, alpha=0.6)
axes[1].set_xlabel('Numero de Habitaciones', fontsize = 12)
axes[1].set_ylabel('Precio (miles $)', fontsize=12)
axes[1].set_title('Habitaciones vs Precio', fontweight='bold')
axes[1].grid(True,alpha=0.3)

axes[2].scatter(df['antiguedad_anos'], df['precio_miles'], color='red', s=100, alpha=0.6)
axes[2].set_xlabel('Antiguedad (Años)', fontsize=12)
axes[2].set_ylabel('Precio (miles $)', fontsize=12)
axes[2].set_title('Antiguedad vs Precio', fontweight='bold')
axes[2].grid(True, alpha=0.3)

print("SEPARAR CARACTERÍSTICAS (X) Y OBJETIVO (y)")

y = df['precio_miles']
X = df.drop('precio_miles', axis=1)

print(f"\nForma de X (Características): {X.shape}")
print(f"Forma de y (Objetivo): {y.shape}")

print("\nPrimeras 5 casas:")
print(X.head())
print("\nPrimeros 5 precios:")
print(y.head())

print("DIVIDIR EN ENTRENAMIENTO Y PRUEBA")

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.3, # 30% para prueba
  random_state=42
)

print(f"\nDatos de entrenamiento: {len(X_train)} casas | Datos de prueba: {len(X_test)} casas")

print("\nCasas en ENTRENAMIENTO:")
print(pd.DataFrame({
  'Tamaño': X_train['tamano_m2'].values,
  'Habitaciones': X_train['num_habitaciones'].values,
  'Antiguedad': X_train['antiguedad_anos'].values,
  'Precio': y_train.values
}))

print("\nCasas en PRUEBA (el modelo NO las ve durante entrenamiento)")
print(pd.DataFrame({
  'Tamaño': X_test['tamano_m2'].values,
  'Habitaciones': X_test['num_habitaciones'].values,
  'Antiguedad': X_test['antiguedad_anos'].values,
  'Precio': y_test.values
}))

print("CREAR Y ENTRENAR ÁRBOL DE DECISIÓN")

# Crear arbol con poca profundidad para visualizarlo fácilmente
modelo_tree = DecisionTreeRegressor(
  max_depth=3,     # Solo 3 niveles de profundidad
  min_samples_split=2, # Mínimo 2 muestras para dividir
  min_samples_leaf=1,  #Mínimo 1 muestra de hoja
  random_state=42
)

print("\nConfiguración del Árbol:")
print(f"  - Profundidad máxima: {modelo_tree.max_depth}")
print(f"  - Mínimo para dividir: {modelo_tree.min_samples_split}")
print(f"  - Mínimo en hoja: {modelo_tree.min_samples_leaf}")


# Entrenamiento del árbol

modelo_tree.fit(X_train,y_train)
print("\nÁrbol Entrenado Correctamente")

# VISUALIZACIÓN DEL ÁRBOL
plt.figure(figsize=(20,10))

plot_tree(
  modelo_tree,
  feature_names=['Tamaño','Habitaciones','Antiguedad'],
  filled=True,
  rounded=True,
  fontsize=10
)

plt.title('Árbol de Decisión - Predicción de Precios', fontsize=16,fontweight='bold')
plt.show()

# PREDICCIONES

# Predecir datos de entrenamiento
y_train_pred = modelo_tree.predict(X_train)

# Predecir datos de prueba
y_test_pred = modelo_tree.predict(X_test)

print("\nPREDECCIONES EN DATOS DE PRUEBA:")

comparacion_test = pd.DataFrame({
  'Tamaño (m^2)': X_test['tamano_m2'].values,
  'Habitaciones': X_test['num_habitaciones'].values,
  'Precio Real': y_test.values,
  'Precio Predicho': y_test_pred,
  'Error': y_test.values - y_test_pred,
  'Error %': ((y_test.values - y_test_pred) / y_test.values *100).round(1)
})

print(comparacion_test.to_string(index=False))

# EVALUAR MODELO

# Metricas de ENTRENAMIENTO
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train,y_train_pred)
r2_train = r2_score(y_train,y_train_pred)

# Métricas de PRUEBA
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test,y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("MÉTRICAS DE ENTRENAMIENTO")
print(f'  MSE: {mse_train:.2f}')
print(f"  RMSE: ${rmse_train:.2f}k (error promedio)")
print(f"  MAE: ${mae_train:.2f}k")
print(f"  R2: {r2_train:.4f} ({r2_train*100:.2f}% varianza explicada)")

print("\nMÉTRICAS DE PRUEBA")
print(f"  MSE: {mse_test:.2f}")
print(f"  RMSE: ${rmse_test:.2f}k (error promedio)")
print(f"  MAE: ${mae_test:.2f}k")
print(f"  R2: {r2_test:.4f} ({r2_test*100:.2f}% varianza explicada)")

print("\nANÁLISIS DE OVERFITTING:")
diferencia = r2_train - r2_test
print(f"Diferencia r2_train - r2_test = {diferencia:.2f}")
if diferencia < 0.05:
  print("Modelo generaliza bien (diferencia < 0.05)")
elif diferencia < 0.15:
  print("Posible ligero overfitting (diferencia 0.05-0.15)")
else:
  print("Overfitting significativo (diferencia > 0.15)")

print("IMPORTANCIA DE CARACTERÍSTICAS")

importancia = modelo_tree.feature_importances_
caracteristicas = ['Tamaño', 'Habitaciones', 'Antiguedad']

df_importancia = pd.DataFrame({
  'Característica': caracteristicas,
  'Importancia': importancia,
  'Importancia %': (importancia*100).round(2)
}).sort_values('Importancia', ascending=False)

print("Caracterísitcas que usa más el árbol para decidir")
print(df_importancia.to_string(index=False))

plt.figure(figsize=(10,6))

plt.barh(df_importancia['Característica'],df_importancia['Importancia %'], color='steelblue', edgecolor='black')
plt.xlabel('Importancia (%)', fontsize=12)
plt.title('Importancia de Características en el Árbol', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nINTERPRETACIÓN:")
print(f"  - {df_importancia.iloc[0]['Característica']} es la MÁS importante"
      f"  - {df_importancia.iloc[0]['Importancia %']:.1f}%")
print(f"  - El árbol hace la mayoría de las decisiones basándose en esta característica")

print("VISUALIZACIÓN DE PREDICCIONES")

fig,axes=plt.subplots(1,2,figsize=(16,6))

# Gráfico 1: Real vs Predicho
axes[0].scatter(y_test,y_test_pred, s=150, alpha=0.6, edgecolors='black')
axes[0].set_xlabel('Precio Real ($k)', fontsize=12)
axes[0].set_ylabel('Precio Predicho ($k)', fontsize=12)
axes[0].set_title('Real vs Predicho', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True,alpha=0.3)

# Agregar etiquetas a cada punto
for i, (real,pred) in enumerate(zip(y_test,y_test_pred)):
  axes[0].annotate(f'{i+1}', (real, pred), fontsize=9)

# Gráfico 2: Residuos
residuos = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuos, s=150, alpha=0.6, color='purple', edgecolors='black')
axes[1].axhline(y=0,color='red',linestyle='--',lw=2)
axes[1].set_xlabel('Precio Predicho ($k)',fontsize=12)
axes[1].set_ylabel('Residuos (Error)', fontsize=12)
axes[1].set_title('Análisis de Residuos', fontweight='bold', fontsize=14)
axes[1].grid(True,alpha=0.3)

plt.tight_layout()
plt.show()

print(f"""
MODELO ENTRENADO: Decision Tree (Árbol de Decisión)
DATOS: 20 casas (14 entrenamiento, 6 prueba)
CARACTERÍSTICAS: Tamaño, Habitaciones, Antigüedad

 DESEMPEÑO:
  - RMSE en Test: ${rmse_test:.2f}k (error promedio)
  - R² en Test: {r2_test:.4f} ({r2_test*100:.1f}% varianza explicada)
  - Característica más importante: {df_importancia.iloc[0]['Característica']}

 VENTAJAS DEL ÁRBOL:
  ✓ Fácil de interpretar (puedes ver las decisiones)
  ✓ No necesita escalado de datos
  ✓ Maneja relaciones no lineales
  ✓ Rápido de entrenar

 DESVENTAJAS:
  ✗ Puede sobreajustar fácilmente
  ✗ Inestable (pequeños cambios → árbol diferente)
  ✗ No predice fuera del rango visto

 CUÁNDO USAR:
  - Cuando necesitas interpretabilidad
  - Como baseline para comparar con modelos más complejos
  - Cuando tienes datos categóricos mezclados con numéricos
""")
