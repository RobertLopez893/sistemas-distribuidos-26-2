# Práctica 1: Multiprocesamiento
# López Reyes José Roberto. 7CM1.

import time
import multiprocessing
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# --- Funciones del Worker ---

def instanciar_modelo(nombre, params):
    # Factory para instanciar el modelo según el nombre
    if nombre == 'SVM':
        return SVC(**params)
    elif nombre == 'Random Forest':
        return RandomForestClassifier(**params)
    elif nombre == 'Regresión Logística':
        return LogisticRegression(**params, max_iter=1000)
    elif nombre == 'Árbol de Decisión':
        return DecisionTreeClassifier(**params)
    elif nombre == 'Naive Bayes':
        return GaussianNB(**params)
    elif nombre == 'MLP (Red Neuronal)':
        return MLPClassifier(**params, max_iter=1000)
    return None


def tarea_entrenamiento(args):
    # Función principal ejecutada por cada núcleo
    nombre_modelo, parametros, X_train, X_test, y_train, y_test = args
    proc_name = multiprocessing.current_process().name

    print(f"[{proc_name}] Entrenando {nombre_modelo} con params: {parametros}...")
    start = time.time()

    clf = instanciar_modelo(nombre_modelo, parametros)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    end = time.time()
    return {'modelo': nombre_modelo, 'params': parametros, 'accuracy': acc, 'tiempo': end - start}


# --- Proceso Maestro ---

if __name__ == '__main__':
    # Generación de datos (10k muestras)
    print("--- INICIO: Generando Dataset ---")
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_cores = multiprocessing.cpu_count()
    print(f"Núcleos detectados: {n_cores}")
    print("-" * 60)

    # --- FASE 1: Competencia de Modelos ---
    print("\n>>> FASE 1: Selección de Modelo")

    modelos_fase1 = [
        ('SVM', {'kernel': 'rbf'}),
        ('Random Forest', {'n_estimators': 100}),
        ('Regresión Logística', {'C': 1.0}),
        ('Árbol de Decisión', {'criterion': 'gini'}),
        ('Naive Bayes', {}),
        ('MLP (Red Neuronal)', {'hidden_layer_sizes': (50,)})
    ]

    tareas_fase1 = [(nombre, params, X_train, X_test, y_train, y_test) for nombre, params in modelos_fase1]

    ts_inicio_f1 = time.time()

    # Ejecución paralela Fase 1
    with multiprocessing.Pool(processes=n_cores) as pool:
        resultados_f1 = pool.map(tarea_entrenamiento, tareas_fase1)

    ts_fin_f1 = time.time()

    print("\nRESULTADOS FASE 1:")
    print(f"{'Modelo':<25} | {'Accuracy':<10} | {'Tiempo (s)':<10}")
    print("-" * 50)
    for res in resultados_f1:
        print(f"{res['modelo']:<25} | {res['accuracy']:.4f}     | {res['tiempo']:.4f}")

    ganador = max(resultados_f1, key=lambda x: x['accuracy'])
    print(f"\nGANADOR FASE 1: {ganador['modelo']} ({ganador['accuracy']:.4f})")
    print(f"Tiempo Total Fase 1: {ts_fin_f1 - ts_inicio_f1:.4f}s")
    print("-" * 60)

    # --- FASE 2: Optimización (Grid Search) ---
    print(f"\n>>> FASE 2: Optimizando {ganador['modelo']}")

    nombre_ganador = ganador['modelo']
    grid_params = []

    # Definición del Grid según el ganador
    if nombre_ganador == 'SVM':
        grid_params = [{'C': 0.1, 'kernel': 'rbf'}, {'C': 1, 'kernel': 'rbf'},
                       {'C': 10, 'kernel': 'rbf'}, {'C': 1, 'kernel': 'linear'}]
    elif nombre_ganador == 'Random Forest':
        grid_params = [{'n_estimators': 50, 'max_depth': 10}, {'n_estimators': 100, 'max_depth': 10},
                       {'n_estimators': 200, 'max_depth': None}, {'n_estimators': 300, 'max_depth': None}]
    elif nombre_ganador == 'MLP (Red Neuronal)':
        grid_params = [{'hidden_layer_sizes': (50,)}, {'hidden_layer_sizes': (100,)},
                       {'hidden_layer_sizes': (50, 50)}, {'hidden_layer_sizes': (100, 50)}]
    elif nombre_ganador == 'Árbol de Decisión':
        grid_params = [{'max_depth': 5}, {'max_depth': 10}, {'max_depth': 20}, {'max_depth': None}]
    elif nombre_ganador == 'Regresión Logística':
        grid_params = [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}, {'C': 100.0}]
    else:
        grid_params = [{'var_smoothing': 1e-9}, {'var_smoothing': 1e-8}, {'var_smoothing': 1e-5}]

    tareas_fase2 = [(nombre_ganador, params, X_train, X_test, y_train, y_test) for params in grid_params]

    ts_inicio_f2 = time.time()

    # Ejecución paralela Fase 2
    with multiprocessing.Pool(processes=n_cores) as pool:
        resultados_f2 = pool.map(tarea_entrenamiento, tareas_fase2)

    ts_fin_f2 = time.time()

    print("\nRESULTADOS FASE 2 (Variaciones):")
    for res in resultados_f2:
        print(f"Config: {res['params']} -> Acc: {res['accuracy']:.4f}")

    mejor_config = max(resultados_f2, key=lambda x: x['accuracy'])

    print("\n" + "=" * 60)
    print(f"CONCLUSIÓN FINAL: {mejor_config['modelo']}")
    print(f"Mejor Configuración: {mejor_config['params']}")
    print(f"Precisión Final: {mejor_config['accuracy']:.4f}")
    print("=" * 60)
