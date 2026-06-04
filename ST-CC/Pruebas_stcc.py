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

# PASO 2: Co-Clustering Checkerboard con BBAC_I

n_row_clusters = min(8, num_regiones) 
n_col_clusters = min(4, num_tiempos)

# Instanciamos nuestra clase personalizada
# Usamos n_init=10 para no demorar mucho probando, luego súbelo a 200
model = BBAC_I(n_row_clusters=n_row_clusters, 
               n_col_clusters=n_col_clusters, 
               n_init=200, 
               max_iter=2000, 
               tol=1e-6)

model.fit(X)
print(f"Mejor pérdida de I-divergencia alcanzada: {model.best_loss_:.4f}")


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