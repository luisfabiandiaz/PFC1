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

def cabinet_air_quality_baseline(X_matrix, K_dims=10, k_neighbors=5):
    """
    Implementación base de CAbiNet para datos espaciotemporales (Calidad de Aire).
    X_matrix: Array 2D (Filas = Estaciones/Espacio, Columnas = Días/Tiempo).
              Debe contener valores no negativos (ej. concentraciones de NO2).
    """
    # ---------------------------------------------------------
    # PASO 1: Análisis de Correspondencia (CA) y SVD
    # ---------------------------------------------------------
    # 1.1 Convertir a matriz de probabilidad (contingencia)
    N_plus_plus = np.sum(X_matrix)
    P = X_matrix / N_plus_plus
    
    # 1.2 Calcular sumas de filas (r) y columnas (c)
    r = np.sum(P, axis=1)
    c = np.sum(P, axis=0)
    
    # 1.3 Calcular probabilidades esperadas y Residuos de Pearson (S)
    # S_ij = (P_ij - r_i * c_j) / sqrt(r_i * c_j)
    expected = np.outer(r, c)
    S = (P - expected) / np.sqrt(expected)
    
    # 1.4 SVD (Descomposición en Valores Singulares) para reducir dimensionalidad
    svd = TruncatedSVD(n_components=K_dims)
    U = svd.fit_transform(S) # Coordenadas principales de las filas (Estaciones)
    V = svd.components_.T    # Coordenadas de las columnas (Días)
    alpha = svd.singular_values_
    
    # Coordenadas Principales (G, F) y Estándar (Gamma, Phi)
    # Simplificamos asumiendo que U y V ya escalan la varianza para el prototipo
    G_estaciones = U 
    G_dias = V * alpha
    
    # ---------------------------------------------------------
    # PASO 2: Construcción del Grafo Bimodal (Espacio-Tiempo)
    # ---------------------------------------------------------
    # En CAbiNet construimos subgrafos: Espacio-Espacio, Tiempo-Tiempo, y Espacio-Tiempo.
    
    # 2.1 Grafo Estación-Estación (KNN basado en distancia Euclidiana)
    nn_est = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_est.fit(G_estaciones)
    adj_est_est = nn_est.kneighbors_graph(G_estaciones, mode='connectivity')
    
    # 2.2 Grafo Día-Día (KNN basado en distancia Euclidiana)
    nn_dia = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_dia.fit(G_dias)
    adj_dia_dia = nn_dia.kneighbors_graph(G_dias, mode='connectivity')
    
    # 2.3 Grafo Estación-Día (Basado en el "Association Ratio" o producto interno)
    # asociation_ratio = np.dot(G_estaciones, Coordenadas_Estandar_Dias.T)
    association_matrix = np.dot(G_estaciones, V.T) 
    
    # Conectamos cada estación con los 'k' días más asociados (y viceversa)
    adj_est_dia = np.zeros_like(association_matrix)
    for i in range(association_matrix.shape[0]):
        top_k_indices = np.argsort(association_matrix[i])[-k_neighbors:]
        adj_est_dia[i, top_k_indices] = 1
        
    # 2.4 Ensamblar la matriz de adyacencia global (Estaciones + Días)
    n_est = G_estaciones.shape[0]
    n_dias = G_dias.shape[0]
    n_total = n_est + n_dias
    
    global_adj = np.zeros((n_total, n_total))
    global_adj[:n_est, :n_est] = adj_est_est.toarray()
    global_adj[n_est:, n_est:] = adj_dia_dia.toarray()
    global_adj[:n_est, n_est:] = adj_est_dia
    global_adj[n_est:, :n_est] = adj_est_dia.T # Simetría
    
    # ---------------------------------------------------------
    # PASO 3: Co-clustering (Leiden)
    # ---------------------------------------------------------
    # Convertimos a grafo de igraph

    # --- LÍNEA AÑADIDA
    # Forzamos simetría: si A conecta a B, B conecta a A
    global_adj = np.maximum(global_adj, global_adj.T) 
    
    # Ahora ya puedes crear el grafo sin errores
    G = ig.Graph.Adjacency((global_adj > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    
    # Aplicamos algoritmo Leiden para encontrar los clústeres conjuntos
    particion = la.find_partition(G, la.ModularityVertexPartition)
    clusters = particion.membership
    
    # ---------------------------------------------------------
    # PASO 4: Visualización biMAP (UMAP)
    # ---------------------------------------------------------
    # UMAP incrusta nodos (Estaciones y Días) en 2D para visualización conjunta
    reducer = umap.UMAP(n_neighbors=k_neighbors, min_dist=0.1, metric='jaccard')
    embedding_2d = reducer.fit_transform(global_adj)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Separar coordenadas
    emb_est = embedding_2d[:n_est]
    emb_dias = embedding_2d[n_est:]
    
    # Extraer clústeres
    clust_est = clusters[:n_est]
    clust_dias = clusters[n_est:]
    
    # Graficar Días (Puntos pequeños)
    plt.scatter(emb_dias[:, 0], emb_dias[:, 1], c=clust_dias, cmap='tab20', 
                s=20, alpha=0.6, label='Días (Tiempo)')
    
    # Graficar Estaciones (Estrellas más grandes con borde)
    plt.scatter(emb_est[:, 0], emb_est[:, 1], c=clust_est, cmap='tab20', 
                s=150, marker='*', edgecolors='black', label='Estaciones (Espacio)')
    
    plt.title("biMAP Espaciotemporal: Calidad de Aire")
    plt.legend()
    plt.show()
    
    return clusters, embedding_2d

# --- Ejemplo de Uso ---
# 1. Cargar los datos
# Usamos sep=';' por el formato de tu CSV y decimal=',' para leer bien los números
df = pd.read_csv("D:/UCSP/Proyecto_final_de_carrera/Posibles_papers/a_implementar/AirQualityPACA_Data-master/data_polmet.csv", sep=";", decimal=",")

# Opcional: Asegurar que 'date' sea formato datetime para que se ordene cronológicamente
df['date'] = pd.to_datetime(df['date'])

# 2. Pivotear la tabla para armar la Matriz Espaciotemporal
# Filas = 'cp' (Espacio), Columnas = 'date' (Tiempo), Valores = 'max_NO2max'
matriz_df = df.pivot(index='cp', columns='date', values='max_NO2max')

# 3. Imputar valores nulos
# Siguiendo tu documento: "los valores nulos derivados de ausencias de monitoreo fueron imputados con ceros"
matriz_df = matriz_df.fillna(0) 

# 4. Extraer la matriz pura para el algoritmo y guardar las etiquetas
X_matrix_real = matriz_df.values
estaciones_nombres = matriz_df.index.tolist()
dias_nombres = matriz_df.columns.astype(str).tolist()

print(f"¡Matriz lista! Dimensión: {X_matrix_real.shape[0]} zonas espaciales x {X_matrix_real.shape[1]} días.")

# 5. Ejecutar tu algoritmo con la data real
clusters, proyeccion = cabinet_air_quality_baseline(X_matrix_real)


# 1. Separar los clústeres devueltos por el algoritmo
n_estaciones = len(estaciones_nombres)
clust_est = clusters[:n_estaciones]
clust_dias = clusters[n_estaciones:]

# 2. Reordenar la matriz original (para el panel "ANTES")
# Obtenemos los índices que ordenarían las estaciones y días por su número de clúster
idx_est_ordenados = np.argsort(clust_est)
idx_dias_ordenados = np.argsort(clust_dias)

# Aplicamos el reordenamiento a la matriz (esto agrupa visualmente los bloques)
matriz_reordenada = X_matrix_real[idx_est_ordenados, :][:, idx_dias_ordenados]

# 3. Configurar el lienzo para la comparación (1 fila, 2 columnas)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- PANEL IZQUIERDO: El "Antes" (Matriz Abstracta) ---
sns.heatmap(matriz_reordenada, cmap="viridis", ax=axes[0], 
            cbar_kws={'label': 'Niveles máximos de NO2'}, xticklabels=False, yticklabels=False)
axes[0].set_title("ANTES: Partición Abstracta\n(Matriz reordenada por clústeres pero de difícil validación)", fontsize=14)
axes[0].set_xlabel("Días (Ordenados por clúster)")
axes[0].set_ylabel("Códigos Postales (Ordenados por clúster)")

# --- PANEL DERECHO: El "Después" (El biMAP de CAbiNet) ---
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

# Ajustar layout y mostrar
plt.tight_layout()
plt.show()