import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralBiclustering, KMeans

# PASO 1: Carga y Preparación de Datos Reales

df = pd.read_csv('D:/UCSP\Proyecto_final_de_carrera/Posibles_papers/a_implementar/AirQualityPACA_Data-master/data_polmet.csv', sep=';', decimal=',')

# 1.2 Crear la matriz de co-ocurrencia (Pivot Table)
# Filas: Regiones ('cp')
# Columnas: Tiempos ('date')
# Valores: Escoge una variable, por ejemplo 'max_NO2max'
matriz_df = df.pivot_table(index='cp', columns='date', values='max_NO2max', aggfunc='mean')

# 1.3 Limpieza de datos
matriz_df = matriz_df.fillna(0)

X = matriz_df.values
nombres_regiones = matriz_df.index.astype(str).tolist()
nombres_tiempos = matriz_df.columns.astype(str).tolist()

num_regiones, num_tiempos = X.shape
print(f"Matriz creada: {num_regiones} regiones x {num_tiempos} tiempos.")

# PASO 2: Spectral Biclustering

n_row_clusters = min(8, num_regiones) 
n_col_clusters = min(4, num_tiempos)

model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters), random_state=42)
model.fit(X)

# PASO 3: Extracción de características y K-means
block_features = []

for r in range(n_row_clusters):
    for c in range(n_col_clusters):
        filas_bloque = np.where(model.row_labels_ == r)[0]
        cols_bloque = np.where(model.column_labels_ == c)[0]
        
        datos_bloque = X[np.ix_(filas_bloque, cols_bloque)]
        
        if datos_bloque.size > 0:
            media = np.mean(datos_bloque)
            std = np.std(datos_bloque)
        else:
            media, std = 0, 0
            
        block_features.append([media, std])

block_features = np.array(block_features)

# Aplicar K-means
k_optimo = min(4, len(block_features))
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init='auto')
final_labels = kmeans.fit_predict(block_features)


# PASO 4: Preparar las matrices para visualización

row_indices = np.argsort(model.row_labels_)
col_indices = np.argsort(model.column_labels_)

# ANTES
X_reordered = X[row_indices, :][:, col_indices]

# DESPUÉS
X_kmeans = np.zeros_like(X, dtype=float)
block_idx = 0

for r in range(n_row_clusters):
    for c in range(n_col_clusters):
        filas = np.where(model.row_labels_ == r)[0]
        cols = np.where(model.column_labels_ == c)[0]
        for f in filas:
            for cl in cols:
                X_kmeans[f, cl] = final_labels[block_idx]
        block_idx += 1

X_kmeans_reordered = X_kmeans[row_indices, :][:, col_indices]

# PASO 5: Graficar

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico 1
sns.heatmap(X_reordered, cmap="viridis", ax=axes[0], cbar_kws={'label': 'Niveles de NO2'})
axes[0].set_title("ANTES: Co-Clustering Checkerboard\n(Valores crudos agrupados)")
axes[0].set_xlabel("Días")
axes[0].set_ylabel("Códigos Postales (Regiones)")

if num_tiempos > 20: axes[0].set_xticks([]) 
if num_regiones > 20: axes[0].set_yticks([])

# Gráfico 2
sns.heatmap(X_kmeans_reordered, cmap="Set1", ax=axes[1], cbar_kws={'label': 'Patrón K-means'})
axes[1].set_title("DESPUÉS: Refinamiento con K-means\n(Discretizado en comportamientos)")
axes[1].set_xlabel("Días")
axes[1].set_ylabel("Códigos Postales (Regiones)")

if num_tiempos > 20: axes[1].set_xticks([])
if num_regiones > 20: axes[1].set_yticks([])

plt.tight_layout()
plt.show()