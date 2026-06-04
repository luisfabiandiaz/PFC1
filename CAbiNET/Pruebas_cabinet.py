import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import leidenalg as la
import igraph as ig

def cabinet_air_quality_baseline(X_matrix, K_dims=10, k_neighbors=5, jaccard_threshold=0.15):
    """
    Implementación de CAbiNet con Grafo SNN intermedio.
    X_matrix: Array 2D (Filas = Estaciones/Espacio, Columnas = Días/Tiempo).
    jaccard_threshold: Umbral para la poda de aristas en el grafo SNN.
    """
    # PASO 1: Análisis de Correspondencia (CA) y SVD
    N_plus_plus = np.sum(X_matrix)
    P = X_matrix / N_plus_plus
    
    r = np.sum(P, axis=1)
    c = np.sum(P, axis=0)
    
    expected = np.outer(r, c)
    S = (P - expected) / np.sqrt(expected)
    
    svd = TruncatedSVD(n_components=K_dims)
    U = svd.fit_transform(S) 
    V = svd.components_.T    
    alpha = svd.singular_values_
    
    G_estaciones = U 
    G_dias = V * alpha
    
    # PASO 2: Construcción del Grafo Bimodal (Espacio-Tiempo) inicial
    nn_est = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_est.fit(G_estaciones)
    adj_est_est = nn_est.kneighbors_graph(G_estaciones, mode='connectivity')
    
    nn_dia = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_dia.fit(G_dias)
    adj_dia_dia = nn_dia.kneighbors_graph(G_dias, mode='connectivity')
    
    association_matrix = np.dot(G_estaciones, V.T) 
    
    adj_est_dia = np.zeros_like(association_matrix)
    for i in range(association_matrix.shape[0]):
        top_k_indices = np.argsort(association_matrix[i])[-k_neighbors:]
        adj_est_dia[i, top_k_indices] = 1
        
    n_est = G_estaciones.shape[0]
    n_dias = G_dias.shape[0]
    n_total = n_est + n_dias
    
    global_adj = np.zeros((n_total, n_total))
    global_adj[:n_est, :n_est] = adj_est_est.toarray()
    global_adj[n_est:, n_est:] = adj_dia_dia.toarray()
    global_adj[:n_est, n_est:] = adj_est_dia
    global_adj[n_est:, :n_est] = adj_est_dia.T 
    
    # Forzamos simetría del grafo k-NN base
    global_adj = np.maximum(global_adj, global_adj.T) 
    
    # PASO 2.5: Transformación a Grafo SNN (Similitud de Jaccard)
    # Incluimos los self-loops temporalmente para que un nodo sea parte de su propio vecindario
    np.fill_diagonal(global_adj, 1.0)
    
    # Calculamos la intersección de vecindarios (A @ A.T)
    intersection = np.dot(global_adj, global_adj.T)
    degrees = global_adj.sum(axis=1)
    
    # Calculamos la unión usando broadcasting
    union = degrees[:, np.newaxis] + degrees[np.newaxis, :] - intersection
    
    # Similitud de Jaccard = Intersección / Unión
    with np.errstate(divide='ignore', invalid='ignore'):
        snn_jaccard = np.where(union > 0, intersection / union, 0.0)
    
    # Poda (Thresholding) de las aristas débiles
    snn_jaccard[snn_jaccard < jaccard_threshold] = 0.0
    
    # Removemos la diagonal para evitar que los auto-bucles afecten a Leiden
    np.fill_diagonal(snn_jaccard, 0.0)
    
    # PASO 3: Co-clustering (Leiden) sobre el grafo ponderado SNN
    # Usamos Weighted_Adjacency para que igraph asimile la matriz con ponderaciones reales
    G = ig.Graph.Weighted_Adjacency(snn_jaccard.tolist(), mode=ig.ADJ_UNDIRECTED)
    
    # Extraemos los pesos de las aristas para inyectarlos en Leiden
    weights = G.es["weight"]
    particion = la.find_partition(G, la.ModularityVertexPartition, weights=weights)
    clusters = particion.membership

    # PASO 4: Visualización biMAP (UMAP)
    # Convertimos la matriz de similitud SNN en una matriz de distancias (Distancia = 1 - Similitud)
    dist_matrix = 1.0 - snn_jaccard
    np.fill_diagonal(dist_matrix, 0.0) # La distancia hacia sí mismo es 0
    
    # Configuramos UMAP para recibir directamente la matriz de distancias precalculada
    reducer = umap.UMAP(n_neighbors=k_neighbors, min_dist=0.1, metric='precomputed')
    embedding_2d = reducer.fit_transform(dist_matrix)
    
    # Plotting (Sin cambios)
    plt.figure(figsize=(10, 8))
    
    emb_est = embedding_2d[:n_est]
    emb_dias = embedding_2d[n_est:]
    
    clust_est = clusters[:n_est]
    clust_dias = clusters[n_est:]
    
    plt.scatter(emb_dias[:, 0], emb_dias[:, 1], c=clust_dias, cmap='tab20', 
                s=20, alpha=0.6, label='Días (Tiempo)')
    
    plt.scatter(emb_est[:, 0], emb_est[:, 1], c=clust_est, cmap='tab20', 
                s=150, marker='*', edgecolors='black', label='Estaciones (Espacio)')
    
    plt.title("biMAP Espaciotemporal: Calidad de Aire (Con optimización SNN)")
    plt.legend()
    plt.show()
    
    return clusters, embedding_2d


df = pd.read_csv("D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/AirQualityPACA_Data-master/data_polmet.csv", sep=";", decimal=",")

df['date'] = pd.to_datetime(df['date'])

# Pivotear la tabla para armar la Matriz Espaciotemporal
# Filas = 'cp' (Espacio), Columnas = 'date' (Tiempo), Valores = 'max_NO2max'
matriz_df = df.pivot(index='cp', columns='date', values='max_NO2max')


# 1. Prioridad: Imputar con la media de cada estación (fila) a lo largo del tiempo.
# Esto asume que el comportamiento en un día faltante se asemeja al promedio de ese sensor.
matriz_df = matriz_df.apply(lambda row: row.fillna(row.mean()), axis=1)

# 2. Respaldo de seguridad: Si alguna estación tuviera 100% de NaNs (sin datos), 
# la media de la fila también sería NaN. Para evitar que el SVD falle, 
# imputamos cualquier NaN restante con la media general de toda la matriz.
if matriz_df.isna().sum().sum() > 0:
    media_global = matriz_df.mean().mean()
    matriz_df = matriz_df.fillna(media_global)

# Extraer la matriz pura para el algoritmo y guardar las etiquetas
X_matrix_real = matriz_df.values
estaciones_nombres = matriz_df.index.tolist()
dias_nombres = matriz_df.columns.astype(str).tolist()

print(f"¡Matriz lista! Dimensión: {X_matrix_real.shape[0]} zonas espaciales x {X_matrix_real.shape[1]} días.")

# Ejecutar tu algoritmo con la data real
clusters, proyeccion = cabinet_air_quality_baseline(X_matrix_real)


# Separar los clústeres devueltos por el algoritmo
n_estaciones = len(estaciones_nombres)
clust_est = clusters[:n_estaciones]
clust_dias = clusters[n_estaciones:]

# Reordenar la matriz original (para el panel "ANTES")
# Obtenemos los índices que ordenarían las estaciones y días por su número de clúster
idx_est_ordenados = np.argsort(clust_est)
idx_dias_ordenados = np.argsort(clust_dias)

# Aplicamos el reordenamiento a la matriz (esto agrupa visualmente los bloques)
matriz_reordenada = X_matrix_real[idx_est_ordenados, :][:, idx_dias_ordenados]

# Configurar el lienzo para la comparación (1 fila, 2 columnas)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# PANEL IZQUIERDO:
sns.heatmap(matriz_reordenada, cmap="viridis", ax=axes[0], 
            cbar_kws={'label': 'Niveles máximos de NO2'}, xticklabels=False, yticklabels=False)
axes[0].set_title("ANTES: Partición Abstracta\n(Matriz reordenada por clústeres pero de difícil validación)", fontsize=14)
axes[0].set_xlabel("Días (Ordenados por clúster)")
axes[0].set_ylabel("Códigos Postales (Ordenados por clúster)")

#PANEL DERECHO: 
emb_est = proyeccion[:n_estaciones]
emb_dias = proyeccion[n_estaciones:]

# Graficar Días
axes[1].scatter(emb_dias[:, 0], emb_dias[:, 1], c=clust_dias, cmap='tab20', 
                s=30, alpha=0.5, label='Días (Tiempo)')
# Graficar Estaciones
axes[1].scatter(emb_est[:, 0], emb_est[:, 1], c=clust_est, cmap='tab20', 
                s=250, marker='*', edgecolors='black', linewidths=1.5, label='Estaciones (Espacio)')

axes[1].set_title("DESPUÉS: Proyección Conjunta 2D (biMAP)\n(Patrones solapados revelados interactiva/visualmente)", fontsize=14)
axes[1].legend()

plt.tight_layout()
plt.show()