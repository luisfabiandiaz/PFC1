import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralBiclustering, KMeans

class BBAC_I:
    def __init__(self, n_row_clusters, n_col_clusters, n_init=10, max_iter=2000, tol=1e-6):
        """
        Implementación de Bregman Block Average Co-clustering con I-divergencia.
        Nota: n_init se ha puesto en 10 por defecto para pruebas rápidas. 
        Súbelo a 200 para la corrida final del paper.
        """
        self.k = n_row_clusters
        self.l = n_col_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        
        self.row_labels_ = None
        self.column_labels_ = None
        self.best_loss_ = np.inf

    def _calculate_block_averages(self, X, row_labels, col_labels):
        # Calcula la matriz de promedios M de tamaño (k, l)
        M = np.zeros((self.k, self.l))
        for u in range(self.k):
            for v in range(self.l):
                rows_u = (row_labels == u)
                cols_v = (col_labels == v)
                # Solo calculamos si el bloque no está vacío
                if np.any(rows_u) and np.any(cols_v):
                    M[u, v] = np.mean(X[np.ix_(rows_u, cols_v)])
                else:
                    M[u, v] = 1e-9 # Prevención de división por cero
        return M

    def _update_rows(self, X, row_labels, col_labels):
        M = self._calculate_block_averages(X, row_labels, col_labels)
        
        # Vectorización para comparar cada fila con los k perfiles posibles
        expected_rows = M[:, col_labels] # Forma: (k, n)
        
        # Expandimos dimensiones para cálculo matricial rápido
        X_expanded = X[:, None, :]                  # Forma: (m, 1, n)
        expected_expanded = expected_rows[None, :, :] # Forma: (1, k, n)
        
        # Fórmula de I-divergencia: x * log(x/y) - x + y
        div = X_expanded * np.log(X_expanded / expected_expanded) - X_expanded + expected_expanded
        distances = np.sum(div, axis=2) # Sumamos el error en las n columnas -> (m, k)
        
        # Asignamos cada fila al cluster k con menor divergencia
        return np.argmin(distances, axis=1)

    def _update_cols(self, X, row_labels, col_labels):
        M = self._calculate_block_averages(X, row_labels, col_labels)
        
        expected_cols = M[row_labels, :] # Forma: (m, l)
        
        X_expanded = X[:, :, None]                  # Forma: (m, n, 1)
        expected_expanded = expected_cols[:, None, :] # Forma: (m, 1, l)
        
        div = X_expanded * np.log(X_expanded / expected_expanded) - X_expanded + expected_expanded
        distances = np.sum(div, axis=0) # Sumamos el error en las m filas -> (n, l)
        
        return np.argmin(distances, axis=1)

    def _calculate_loss(self, X, row_labels, col_labels):
        M = self._calculate_block_averages(X, row_labels, col_labels)
        # Reconstruimos la matriz promediada completa
        X_hat = M[row_labels[:, None], col_labels]
        
        div = X * np.log(X / X_hat) - X + X_hat
        return np.sum(div)

    def fit(self, X):
        # Sumamos un epsilon pequeño para evitar logaritmos de 0
        epsilon = 1e-9
        X_safe = X + epsilon
        
        for init_run in range(self.n_init):
            # 1. Inicialización aleatoria
            current_row_labels = np.random.randint(0, self.k, size=X.shape[0])
            current_col_labels = np.random.randint(0, self.l, size=X.shape[1])
            
            loss_history = []
            
            for iteration in range(self.max_iter):
                # 2. Actualizar etiquetas iterativamente
                new_row_labels = self._update_rows(X_safe, current_row_labels, current_col_labels)
                new_col_labels = self._update_cols(X_safe, new_row_labels, current_col_labels)
                
                # 3. Calcular pérdida
                current_loss = self._calculate_loss(X_safe, new_row_labels, new_col_labels)
                loss_history.append(current_loss)
                
                # 4. Verificar convergencia
                if len(loss_history) > 1:
                    loss_change = loss_history[-2] - loss_history[-1]
                    if abs(loss_change) < self.tol:
                        break # Convergencia alcanzada
                        
                current_row_labels = new_row_labels
                current_col_labels = new_col_labels
                
            # Guardamos el mejor modelo de todas las inicializaciones
            if loss_history[-1] < self.best_loss_:
                self.best_loss_ = loss_history[-1]
                self.row_labels_ = current_row_labels
                self.column_labels_ = current_col_labels
                
        return self



# PASO 1: Carga y Preparación de Datos Reales

df = pd.read_csv("D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/DATA_sintetica/experimento_base_4x5.csv", 
                 sep=";", 
                 dtype={'cp': str}, 
                 low_memory=False)


# 1.2 Crear la matriz de co-ocurrencia (Pivot Table)
matriz_df = df.pivot_table(index='cp', columns='date', values='valor_sintetico', aggfunc='mean')

# 1.3 Imputar con la media de cada estación (fila) a lo largo del tiempo.
matriz_df = matriz_df.apply(lambda row: row.fillna(row.mean()), axis=1)

#Extraer el Ground Truth alineado exactamente con el orden del pivoteo
cp_to_gt = df.set_index('cp')['gt_espacial'].to_dict()
filas_real = np.array([cp_to_gt[cp] for cp in matriz_df.index])

date_to_gt = df.set_index('date')['gt_temporal'].to_dict()
columnas_real = np.array([date_to_gt[str(dt)] for dt in matriz_df.columns])

# 1.3.2. Respaldo de seguridad: Si alguna estación tuviera 100% de NaNs.
if matriz_df.isna().sum().sum() > 0:
    media_global = matriz_df.mean().mean()
    matriz_df = matriz_df.fillna(media_global)

print(f"Valores nulos después de imputar: {matriz_df.isna().sum().sum()}")

# Preparar variables para el modelo
X = matriz_df.values
nombres_regiones = matriz_df.index.astype(str).tolist()
nombres_tiempos = matriz_df.columns.astype(str).tolist()

num_regiones, num_tiempos = X.shape
print(f"Matriz creada: {num_regiones} regiones x {num_tiempos} tiempos.")

# PASO 2: Co-Clustering Checkerboard con BBAC_I

n_row_clusters = min(10, num_regiones) 
n_col_clusters = min(12, num_tiempos)

model = BBAC_I(n_row_clusters=n_row_clusters, 
               n_col_clusters=n_col_clusters, 
               n_init=20, 
               max_iter=2000, 
               tol=1e-6)

model.fit(X)
print(f"Mejor pérdida de I-divergencia alcanzada: {model.best_loss_:.4f}")

# PASO 3: Extracción de características de la matriz comprimida M para K-means

M_optima = model._calculate_block_averages(X, model.row_labels_, model.column_labels_)
block_features = []

# Iteramos estrictamente sobre la cuadrícula de co-clusters (k x l)
for r in range(n_row_clusters):
    for c in range(n_col_clusters):
        # La media es directamente el valor representativo del bloque en M
        media = M_optima[r, c]
        
        # Para la desviación estándar, calculamos la dispersión de los elementos 
        # de la matriz reconstruida pertenecientes a este bloque.
        filas_bloque = np.where(model.row_labels_ == r)[0]
        cols_bloque = np.where(model.column_labels_ == c)[0]
        
        if len(filas_bloque) > 0 and len(cols_bloque) > 0:
            # Reconstruimos la porción aproximada del bloque C(R_hat, T_hat)
            bloque_comprimido = np.full((len(filas_bloque), len(cols_bloque)), media)
            std = np.std(bloque_comprimido)                                         
        else:
            std = 0
            
        block_features.append([media, std])

block_features = np.array(block_features)

print(f"Matriz de características generada para K-means: {block_features.shape} (debe ser {n_row_clusters * n_col_clusters} x 2)")

# Aplicar K-means
k_optimo = min(4, len(block_features))
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init='auto')
final_labels = kmeans.fit_predict(block_features)

# PASO 4: Ordenamiento Semantico de clusteres y reconstrucción de la aatriz

# 4.1 Obtener los centroides de K-means
centroides = kmeans.cluster_centers_

# 4.2 Encontrar el orden correcto basado unicamente en la primera columna
indices_ordenados_por_media = np.argsort(centroides[:, 0])

# 4.3 Crear un diccionario de mapeo:
# El clúster con menor media recibirá el valor 0 (Very Low), el mayor recibirá 3 (High)
mapeo_semantico = {id_original: nivel_ordenado for nivel_ordenado, id_original in enumerate(indices_ordenados_por_media)}

# 4.4 Aplicar el mapeo a nuestras etiquetas de bloques para ordenarlas
final_labels_ordered = np.array([mapeo_semantico[label] for label in final_labels])

# 4.5 Reconstrucción de la matriz discretizada
X_kmeans = np.zeros_like(X, dtype=float)
block_idx = 0

for r in range(n_row_clusters):
    for c in range(n_col_clusters):
        filas = np.where(model.row_labels_ == r)[0]
        cols = np.where(model.column_labels_ == c)[0]
        nivel_semantico = final_labels_ordered[block_idx]
        
        for f in filas:
            for cl in cols:
                X_kmeans[f, cl] = nivel_semantico
        block_idx += 1

# Reordenamos ambas matrices para la visualización en checkerboard
row_indices = np.argsort(model.row_labels_)
col_indices = np.argsort(model.column_labels_)

X_reordered = X[row_indices, :][:, col_indices]
X_kmeans_reordered = X_kmeans[row_indices, :][:, col_indices]


# PASO 5: graficacion con escala semantica categorica

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico 1: Valores Crudos Co-Agrupados
sns.heatmap(X_reordered, cmap="viridis", ax=axes[0], cbar_kws={'label': 'Niveles de NO2 (µg/m³)'})
axes[0].set_title("ANTES: Co-Clustering Checkerboard\n(Valores crudos agrupados)")
axes[0].set_xlabel("Días / Horas")
axes[0].set_ylabel("Códigos Postales (Regiones)")

if num_tiempos > 20: axes[0].set_xticks([]) 
if num_regiones > 20: axes[0].set_yticks([])

# Gráfico 2: Refinamiento Discretizado Ordenado (Non-Checkerboard)
# Usamos un mapa de color secuencial y configuramos la barra de color con etiquetas textuales
escalas_color = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"] # Azul (Muy bajo) a Rojo (Alto)
cmap_categorico = sns.color_palette(escalas_color, as_cmap=True)

# Dibujamos el mapa de calor asegurando que los límites numéricos sean estrictamente de 0 a 3
heatmap_kmeans = sns.heatmap(X_kmeans_reordered, 
                             cmap=cmap_categorico, 
                             ax=axes[1], 
                             vmin=-0.5, vmax=3.5,
                             cbar=False) 
cbar = fig.colorbar(heatmap_kmeans.collections[0], ax=axes[1], ticks=[0, 1, 2, 3])
cbar.set_ticklabels(['Very Low', 'Low', 'Medium', 'High'])
cbar.set_label('Estados de Contaminación de NO2')

axes[1].set_title("DESPUÉS: Refinamiento con K-means\n(Discretizado en comportamientos ordenados)")
axes[1].set_xlabel("Días / Horas")
axes[1].set_ylabel("Códigos Postales (Regiones)")

if num_tiempos > 20: axes[1].set_xticks([])
if num_regiones > 20: axes[1].set_yticks([])

plt.tight_layout()
plt.show()