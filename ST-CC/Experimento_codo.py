import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BBAC_I:
    def __init__(self, n_row_clusters, n_col_clusters, n_init=10, max_iter=2000, tol=1e-6):
        self.k = n_row_clusters
        self.l = n_col_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        
        self.row_labels_ = None
        self.column_labels_ = None
        self.best_loss_ = np.inf

    def _calculate_block_averages(self, X, row_labels, col_labels):
        M = np.zeros((self.k, self.l))
        for u in range(self.k):
            for v in range(self.l):
                rows_u = (row_labels == u)
                cols_v = (col_labels == v)
                if np.any(rows_u) and np.any(cols_v):
                    M[u, v] = np.mean(X[np.ix_(rows_u, cols_v)])
                else:
                    M[u, v] = 1e-9 
        return M

    def _update_rows(self, X, row_labels, col_labels):
        M = self._calculate_block_averages(X, row_labels, col_labels)
        expected_rows = M[:, col_labels] 
        X_expanded = X[:, None, :]                  
        expected_expanded = expected_rows[None, :, :] 
        div = X_expanded * np.log(X_expanded / expected_expanded) - X_expanded + expected_expanded
        distances = np.sum(div, axis=2) 
        return np.argmin(distances, axis=1)

    def _update_cols(self, X, row_labels, col_labels):
        M = self._calculate_block_averages(X, row_labels, col_labels)
        expected_cols = M[row_labels, :] 
        X_expanded = X[:, :, None]                  
        expected_expanded = expected_cols[:, None, :] 
        div = X_expanded * np.log(X_expanded / expected_expanded) - X_expanded + expected_expanded
        distances = np.sum(div, axis=0) 
        return np.argmin(distances, axis=1)

    def _calculate_loss(self, X, row_labels, col_labels):
        M = self._calculate_block_averages(X, row_labels, col_labels)
        X_hat = M[row_labels[:, None], col_labels]
        div = X * np.log(X / X_hat) - X + X_hat
        return np.sum(div)

    def fit(self, X):
        epsilon = 1e-9
        X_safe = X + epsilon
        for init_run in range(self.n_init):
            current_row_labels = np.random.randint(0, self.k, size=X.shape[0])
            current_col_labels = np.random.randint(0, self.l, size=X.shape[1])
            loss_history = []
            
            for iteration in range(self.max_iter):
                new_row_labels = self._update_rows(X_safe, current_row_labels, current_col_labels)
                new_col_labels = self._update_cols(X_safe, new_row_labels, current_col_labels)
                
                current_loss = self._calculate_loss(X_safe, new_row_labels, new_col_labels)
                loss_history.append(current_loss)
                
                if len(loss_history) > 1:
                    loss_change = loss_history[-2] - loss_history[-1]
                    if abs(loss_change) < self.tol:
                        break 
                        
                current_row_labels = new_row_labels
                current_col_labels = new_col_labels
                
            if loss_history[-1] < self.best_loss_:
                self.best_loss_ = loss_history[-1]
                self.row_labels_ = current_row_labels
                self.column_labels_ = current_col_labels
                
        return self

# =============================================================================
# PASO 1: Carga rápida de la matriz de datos
# =============================================================================
print("Cargando y preparando el dataset...")

df = pd.read_csv('D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/AirQualityPACA_Data-master/data_polmet.csv', 
                 sep=';', decimal=',', dtype={'cp': str}, low_memory=False)

matriz_df = df.pivot_table(index='cp', columns='date', values='max_NO2max', aggfunc='mean')
matriz_df = matriz_df.apply(lambda row: row.fillna(row.mean()), axis=1)

# archivo_sintetico = "D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/DATA_sintetica/experimento_base_4x5.csv" 
# df = pd.read_csv(archivo_sintetico, sep=';', dtype={'cp': str})

# matriz_df = df.pivot_table(index='cp', columns='date', values='valor_sintetico', aggfunc='mean')
# matriz_df = matriz_df.apply(lambda row: row.fillna(row.mean()), axis=1)


if matriz_df.isna().sum().sum() > 0:
    matriz_df = matriz_df.fillna(matriz_df.mean().mean())

X = matriz_df.values
print(f"Matriz base lista: {X.shape[0]} regiones x {X.shape[1]} tiempos.")

# =============================================================================
# PASO 2: Configuración del Experimento
# =============================================================================
num_experimentos = 5     # 5 corridas independientes
valores_prueba = range(4, 16) # Rango de 4 a 15 clústeres

# Para buscar el codo espacial (k), fijamos el temporal (l)
l_fijo = 4
resultados_k = {exp: [] for exp in range(1, num_experimentos + 1)}

# Para buscar el codo temporal (l), fijamos el espacial (k)
k_fijo = 10
resultados_l = {exp: [] for exp in range(1, num_experimentos + 1)}

# =============================================================================
# PASO 3: Ejecución de Barridos (Grid Search 1D)
# =============================================================================
print("\n[1/2] Iniciando barrido de parámetros para REGIONES (k) [Mantenemos l=4]...")
for k in valores_prueba:
    print(f"  Evaluando k={k}...")
    for exp in range(1, num_experimentos + 1):
        modelo_prueba = BBAC_I(n_row_clusters=k, n_col_clusters=l_fijo, n_init=5, max_iter=10)
        modelo_prueba.fit(X)
        resultados_k[exp].append(modelo_prueba.best_loss_)

print("\n[2/2] Iniciando barrido de parámetros para TIEMPOS (l) [Mantenemos k=10]...")
for l in valores_prueba:
    print(f"  Evaluando l={l}...")
    for exp in range(1, num_experimentos + 1):
        modelo_prueba = BBAC_I(n_row_clusters=k_fijo, n_col_clusters=l, n_init=5, max_iter=10)
        modelo_prueba.fit(X)
        resultados_l[exp].append(modelo_prueba.best_loss_)

# =============================================================================
# PASO 4: Graficación Dual Científica
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
marcadores = ['o', 's', '^', 'D', 'v']

# Gráfica 1: Espacial (k)
for exp in range(1, num_experimentos + 1):
    ax1.plot(valores_prueba, resultados_k[exp], marker=marcadores[exp-1], color=colores[exp-1], 
             linestyle='-', linewidth=1.5, label=f'Exp.{exp}')

ax1.set_title(f'Búsqueda de Codo Espacial (Fijando l={l_fijo})', fontsize=13)
ax1.set_xlabel('Número de region-clusters (k)', fontsize=11)
ax1.set_ylabel('Función Objetivo (I-divergencia)', fontsize=11)
ax1.set_xticks(valores_prueba)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title='Corridas Independientes')

# Gráfica 2: Temporal (l)
for exp in range(1, num_experimentos + 1):
    ax2.plot(valores_prueba, resultados_l[exp], marker=marcadores[exp-1], color=colores[exp-1], 
             linestyle='-', linewidth=1.5, label=f'Exp.{exp}')

ax2.set_title(f'Búsqueda de Codo Temporal (Fijando k={k_fijo})', fontsize=13)
ax2.set_xlabel('Número de time-clusters (l)', fontsize=11)
ax2.set_ylabel('Función Objetivo (I-divergencia)', fontsize=11)
ax2.set_xticks(valores_prueba)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(title='Corridas Independientes')

plt.tight_layout()
plt.savefig('Curva_Codo_Dual_STCC.png', dpi=300)
print("\n¡Experimentos completados! Gráfico guardado como 'Curva_Codo_Dual_STCC.png'.")
plt.show()