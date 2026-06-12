import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg as la

# =============================================================================
# 1. FUNCIÓN DE EVALUACIÓN DE CABINET (VERSIÓN LIGERA PARA GRID SEARCH)
# =============================================================================
def evaluate_cabinet_modularity(X_matrix, K_dims=10, k_neighbors=5, jaccard_threshold=0.15):
    """
    Ejecuta el núcleo de CAbiNet sin visualizaciones y retorna la Modularidad.
    """
    # PASO 1: CA y SVD
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
    
    # PASO 2: Grafo k-NN Bimodal
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
    
    # PASO 3: Red SNN (Shared Nearest Neighbors) y Jaccard
    intersection = np.dot(global_adj, global_adj.T)
    degrees = global_adj.sum(axis=1)
    union = degrees[:, np.newaxis] + degrees[np.newaxis, :] - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        snn_jaccard = np.where(union > 0, intersection / union, 0.0)
    
    snn_jaccard[snn_jaccard < jaccard_threshold] = 0.0
    np.fill_diagonal(snn_jaccard, 0.0)
    
    # PASO 4: Leiden y Modularidad
    # Si la poda fue muy agresiva y el grafo se quedó sin aristas, la modularidad es 0
    if np.sum(snn_jaccard) == 0:
        return 0.0

    G = ig.Graph.Weighted_Adjacency(snn_jaccard.tolist(), mode=ig.ADJ_UNDIRECTED)
    weights = G.es["weight"]
    particion = la.find_partition(G, la.ModularityVertexPartition, weights=weights)
    
    # Retornamos la métrica de calidad de la partición (Modularidad)
    return particion.quality()

# =============================================================================
# 2. CARGA DE LA MATRIZ DE DATOS REALES
# =============================================================================
print("Cargando y preparando el dataset real...")
df = pd.read_csv("D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/DATA_sintetica/experimento_base_4x5.csv", 
                 sep=";", 
                 dtype={'cp': str}, 
                 low_memory=False)


matriz_df = df.pivot(index='cp', columns='date', values='valor_sintetico')
matriz_df = matriz_df.apply(lambda row: row.fillna(row.mean()), axis=1)

if matriz_df.isna().sum().sum() > 0:
    matriz_df = matriz_df.fillna(matriz_df.mean().mean())

X = matriz_df.values
print(f"Matriz lista: {X.shape[0]} regiones x {X.shape[1]} tiempos.")

# =============================================================================
# 3. CONFIGURACIÓN DEL GRID SEARCH (BARRIDO DE PARÁMETROS)
# =============================================================================
# Definimos los valores a explorar
k_neighbors_list = [3, 5, 7, 10, 15]
jaccard_thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]

# Matriz para guardar los resultados
resultados_mod = np.zeros((len(k_neighbors_list), len(jaccard_thresholds)))

print("\nIniciando Grid Search (Esto puede tomar unos minutos)...")
for i, k in enumerate(k_neighbors_list):
    for j, jt in enumerate(jaccard_thresholds):
        print(f"  -> Evaluando k_neighbors={k}, jaccard={jt} ...", end=" ")
        
        try:
            mod_score = evaluate_cabinet_modularity(X, K_dims=10, k_neighbors=k, jaccard_threshold=jt)
            resultados_mod[i, j] = mod_score
            print(f"Modularidad: {mod_score:.4f}")
        except Exception as e:
            # En caso de que una configuración muy extrema rompa el grafo
            resultados_mod[i, j] = 0.0
            print("Error/Grafo vacío.")

# =============================================================================
# 4. GRAFICACIÓN DEL MAPA DE CALOR (HEATMAP)
# =============================================================================
plt.figure(figsize=(10, 7))

# Crear el Heatmap con Seaborn
ax = sns.heatmap(resultados_mod, annot=True, fmt=".4f", cmap="viridis",
                 xticklabels=jaccard_thresholds, yticklabels=k_neighbors_list,
                 cbar_kws={'label': 'Puntuación de Modularidad'})

plt.title('Grid Search CAbiNet: Optimización de Hiperparámetros SNN', fontsize=14, pad=15)
plt.xlabel('Umbral de Jaccard (Poda de aristas débiles)', fontsize=12)
plt.ylabel('Número de Vecinos Iniciales (k_neighbors)', fontsize=12)

# Rotar las etiquetas para que se vean bien
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("grid_search_cabinet_modularity.png", dpi=300)
print("\n¡Búsqueda completada! Gráfico guardado como 'grid_search_cabinet_modularity.png'.")
print("Instrucción: Busca el recuadro con el valor más alto (más amarillo/claro).")
print("Esos son los hiperparámetros óptimos que debes usar en CAbiNet.")
plt.show()