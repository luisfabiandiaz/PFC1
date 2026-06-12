import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg as la
import umap

# 1. CARGA Y PREPARACIÓN DEL DATASET SINTÉTICO
print("Cargando y preparando el dataset sintético...")
archivo_sintetico = "D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/DATA_sintetica/experimento_base_4x5.csv" 
df = pd.read_csv(archivo_sintetico, sep=';', dtype={'cp': str})

matriz_df = df.pivot_table(index='cp', columns='date', values='valor_sintetico', aggfunc='mean')
matriz_df = matriz_df.apply(lambda row: row.fillna(row.mean()), axis=1)

# 2. EXTRACCIÓN DEL GROUND TRUTH

cp_to_gt = df.set_index('cp')['gt_espacial'].to_dict()
filas_real = np.array([cp_to_gt[cp] for cp in matriz_df.index])

date_to_gt = df.set_index('date')['gt_temporal'].to_dict()
columnas_real = np.array([date_to_gt[str(dt)] for dt in matriz_df.columns])

print(f"Matriz preparada: {matriz_df.shape[0]} Espacios x {matriz_df.shape[1]} Tiempos")

# 3. ALGORITMOS 

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

def cabinet_air_quality_baseline(X_matrix, K_dims=10, k_neighbors=5, jaccard_threshold=0.15):
    N_plus_plus = np.sum(X_matrix)
    P = X_matrix / N_plus_plus
    r = np.sum(P, axis=1)
    c = np.sum(P, axis=0)
    expected = np.outer(r, c)
    S = (P - expected) / np.sqrt(expected)
    
    svd = TruncatedSVD(n_components=K_dims, random_state=42)
    U = svd.fit_transform(S) 
    V = svd.components_.T    
    alpha = svd.singular_values_

    U_scaled = U / np.sqrt(r)[:, np.newaxis]
    V_scaled = V / np.sqrt(c)[:, np.newaxis]

    G_estaciones_principal = U_scaled * alpha
    G_dias_estandar = V_scaled
    
    nn_est = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_est.fit(G_estaciones_principal)
    adj_est_est = nn_est.kneighbors_graph(G_estaciones_principal, mode='connectivity')
    
    nn_dia = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_dia.fit(G_dias_estandar)
    adj_dia_dia = nn_dia.kneighbors_graph(G_dias_estandar, mode='connectivity')

    association_matrix = np.dot(G_estaciones_principal, G_dias_estandar.T)
    adj_est_dia = np.zeros_like(association_matrix)

    for i in range(association_matrix.shape[0]):
        top_k_dias = np.argsort(association_matrix[i, :])[-k_neighbors:]
        adj_est_dia[i, top_k_dias] = 1

    for j in range(association_matrix.shape[1]):
        top_k_estaciones = np.argsort(association_matrix[:, j])[-k_neighbors:]
        adj_est_dia[top_k_estaciones, j] = 1
        
    n_est = G_estaciones_principal.shape[0]
    n_dias = G_dias_estandar.shape[0]
    n_total = n_est + n_dias
    
    global_adj = np.zeros((n_total, n_total))
    global_adj[:n_est, :n_est] = adj_est_est.toarray()
    global_adj[n_est:, n_est:] = adj_dia_dia.toarray()
    global_adj[:n_est, n_est:] = adj_est_dia
    global_adj[n_est:, :n_est] = adj_est_dia.T 
    
    global_adj = np.maximum(global_adj, global_adj.T) 
    np.fill_diagonal(global_adj, 1.0)
    
    intersection = np.dot(global_adj, global_adj.T)
    degrees = global_adj.sum(axis=1)
    union = degrees[:, np.newaxis] + degrees[np.newaxis, :] - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        snn_jaccard = np.where(union > 0, intersection / union, 0.0)
    
    snn_jaccard[snn_jaccard < jaccard_threshold] = 0.0
    np.fill_diagonal(snn_jaccard, 0.0)
    
    G = ig.Graph.Weighted_Adjacency(snn_jaccard.tolist(), mode=ig.ADJ_UNDIRECTED)
    weights = G.es["weight"]
    particion = la.find_partition(G, la.ModularityVertexPartition, weights=weights)
    clusters = particion.membership

    dist_matrix = 1.0 - snn_jaccard
    np.fill_diagonal(dist_matrix, 0.0) 
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='precomputed', random_state=42)
    embedding_2d = reducer.fit_transform(dist_matrix)
    
    emb_est = embedding_2d[:n_est]
    emb_dias = embedding_2d[n_est:]
    
    clust_est = clusters[:n_est]
    clust_dias = clusters[n_est:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_dias[:, 0], emb_dias[:, 1], c=clust_dias, cmap='tab20', 
                s=20, alpha=0.6, label='Días (Tiempo)')
    plt.scatter(emb_est[:, 0], emb_est[:, 1], c=clust_est, cmap='tab20', 
                s=150, marker='*', edgecolors='black', label='Estaciones (Espacio)')
    plt.title("biMAP Espaciotemporal: Calidad de Aire (Con optimización SNN)")
    plt.legend()
    plt.show(block=False) 
    plt.pause(2) 

    return np.array(clust_est), np.array(clust_dias), embedding_2d

def calcular_sse_y_reconstruccion(X, row_labels, col_labels):
    """
    Calcula la matriz reconstruida basada en los promedios de bloque 
    y retorna el Error Cuadrático de Reconstrucción (SSE).
    """
    k_clusters = len(np.unique(row_labels))
    l_clusters = len(np.unique(col_labels))
    
    # 1. Crear matriz de promedios de bloque (M)
    M = np.zeros((k_clusters, l_clusters))
    
    # Mapeo de etiquetas únicas a índices (por si acaso los labels saltan números)
    row_mapping = {val: idx for idx, val in enumerate(np.unique(row_labels))}
    col_mapping = {val: idx for idx, val in enumerate(np.unique(col_labels))}
    
    for u in np.unique(row_labels):
        for v in np.unique(col_labels):
            rows_u = (row_labels == u)
            cols_v = (col_labels == v)
            if np.any(rows_u) and np.any(cols_v):
                # Promedio del bloque correspondiente
                M[row_mapping[u], col_mapping[v]] = np.mean(X[np.ix_(rows_u, cols_v)])
    
    # 2. Reconstruir la matriz X_hat
    # Mapeamos cada celda original a su promedio de bloque
    mapped_rows = np.array([row_mapping[r] for r in row_labels])
    mapped_cols = np.array([col_mapping[c] for c in col_labels])
    X_hat = M[mapped_rows[:, None], mapped_cols]
    
    # 3. Calcular SSE 
    sse = np.sum((X - X_hat) ** 2)
    return sse


# 4. EJECUCIÓN DEL EXPERIMENTO Y CÁLCULO DE MÉTRICAS 
print("\nEjecutando algoritmos sobre el escenario sintético...")
X = matriz_df.values 

# Inferimos la cantidad de clusters (k, l) a partir del ground truth
k_espacio = len(np.unique(filas_real))
l_tiempo = len(np.unique(columnas_real))

print("Entrenando ST-CC...")
modelo_stcc = BBAC_I(n_row_clusters=k_espacio, n_col_clusters=l_tiempo, n_init=10)
modelo_stcc.fit(X)

pred_espacio_stcc = modelo_stcc.row_labels_
pred_tiempo_stcc = modelo_stcc.column_labels_

# Métricas ST-CC
ari_espacial_stcc = adjusted_rand_score(filas_real, pred_espacio_stcc)
ari_temporal_stcc = adjusted_rand_score(columnas_real, pred_tiempo_stcc)
sil_espacio_stcc = silhouette_score(X, pred_espacio_stcc, metric='euclidean')
sil_tiempo_stcc = silhouette_score(X.T, pred_tiempo_stcc, metric='euclidean')
sse_stcc = calcular_sse_y_reconstruccion(X, pred_espacio_stcc, pred_tiempo_stcc)

print("Entrenando CAbiNet...")
pred_espacio_cab, pred_tiempo_cab, _ = cabinet_air_quality_baseline(X, k_neighbors=5, jaccard_threshold=0.15)

# Métricas CAbiNet
ari_espacial_cab = adjusted_rand_score(filas_real, pred_espacio_cab)
ari_temporal_cab = adjusted_rand_score(columnas_real, pred_tiempo_cab)
sil_espacio_cab = silhouette_score(X, pred_espacio_cab, metric='euclidean')
sil_tiempo_cab = silhouette_score(X.T, pred_tiempo_cab, metric='euclidean')
sse_cab = calcular_sse_y_reconstruccion(X, pred_espacio_cab, pred_tiempo_cab)


# 5. REPORTE DE RESULTADOS EN CONSOLA
resultados_extendidos = {
    "Algoritmo": ["ST-CC (Línea Base)", "CAbiNet (Propuesta)"],
    "ARI (Espacio)": [ari_espacial_stcc, ari_espacial_cab],
    "ARI (Tiempo)": [ari_temporal_stcc, ari_temporal_cab],
    "Silhouette (Espacio)": [sil_espacio_stcc, sil_espacio_cab],
    "Silhouette (Tiempo)": [sil_tiempo_stcc, sil_tiempo_cab],
    "SSE Global": [sse_stcc, sse_cab]
}

df_reporte = pd.DataFrame(resultados_extendidos)
print("                   REPORTE GLOBAL DE MÉTRICAS (ARI, SILHOUETTE, SSE)")
print(df_reporte.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))


# 6. GENERACIÓN DE GRÁFICOS (DASHBOARD CIENTÍFICO)
print("Generando dashboard de gráficos comparativos...")

algoritmos = ['ST-CC', 'CAbiNet']
x = np.arange(len(algoritmos)) 
width = 0.35 

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#GRÁFICO 1: Adjusted Rand Index (ARI)
axes[0].bar(x - width/2, [ari_espacial_stcc, ari_espacial_cab], width, label='Espacio', color='#1f77b4')
axes[0].bar(x + width/2, [ari_temporal_stcc, ari_temporal_cab], width, label='Tiempo', color='#ff7f0e')
axes[0].set_title('Exactitud del Agrupamiento (ARI)\n¡Más alto es mejor!')
axes[0].set_ylabel('Score ARI')
axes[0].set_xticks(x)
axes[0].set_xticklabels(algoritmos)
axes[0].set_ylim(0, 1.1)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

#GRÁFICO 2: Silhouette Score
axes[1].bar(x - width/2, [sil_espacio_stcc, sil_espacio_cab], width, label='Espacio', color='#2ca02c')
axes[1].bar(x + width/2, [sil_tiempo_stcc, sil_tiempo_cab], width, label='Tiempo', color='#d62728')
axes[1].set_title('Calidad Geométrica de Clústeres (Silhouette)\n¡Más alto es mejor!')
axes[1].set_ylabel('Score Silhouette')
axes[1].set_xticks(x)
axes[1].set_xticklabels(algoritmos)
# El silhouette va de -1 a 1, ajustamos el límite dinámicamente
max_sil = max([sil_espacio_stcc, sil_espacio_cab, sil_tiempo_stcc, sil_tiempo_cab])
axes[1].set_ylim(0, max(1.0, max_sil + 0.2)) 
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

#GRÁFICO 3: Error de Reconstrucción (SSE)
axes[2].bar(algoritmos, [sse_stcc, sse_cab], color='#9467bd', width=0.5)
axes[2].set_title('Error de Reconstrucción de Matriz (SSE)\n¡Más bajo es mejor!')
axes[2].set_ylabel('Suma de Errores al Cuadrado (SSE)')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(ax, rects, is_sse=False):
    for rect in rects:
        height = rect.get_height()
        formato = f'{height:.2f}' if is_sse else f'{height:.4f}'
        ax.annotate(formato,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(axes[0], axes[0].containers[0])
autolabel(axes[0], axes[0].containers[1])
autolabel(axes[1], axes[1].containers[0])
autolabel(axes[1], axes[1].containers[1])
autolabel(axes[2], axes[2].containers[0], is_sse=True)

fig.tight_layout()

nombre_grafico = "comparativa_metricas_globales.png"
plt.savefig(nombre_grafico, dpi=300, bbox_inches='tight')
print(f"-> Gráfico guardado con éxito como '{nombre_grafico}'.")
plt.show()