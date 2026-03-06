import socket
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PORT = 5000


def obtener_configuracion():
    """Menú interactivo tipo MLOps para personalizar el modelo antes de pedir datos."""
    print("=== PLATAFORMA MLOps: CONFIGURACIÓN DE MODELO ===")
    print("1. SVM (Support Vector Machine)")
    print("2. Random Forest")
    print("3. MLP (Red Neuronal)")

    opcion = input("\nElige el modelo a entrenar (1-3): ")

    nombre = ""
    params = {}

    if opcion == '1':
        nombre = 'SVM'
        print("\n--- Configurando SVM ---")
        k = input("Kernel (1: rbf, 2: linear, 3: poly) [Default 1]: ")
        if k == '2':
            params['kernel'] = 'linear'
        elif k == '3':
            params['kernel'] = 'poly'
        else:
            params['kernel'] = 'rbf'

        c = input("Valor de C (ej. 0.1, 1.0, 10.0) [Default 1.0]: ")
        params['C'] = float(c) if c.strip() else 1.0

    elif opcion == '2':
        nombre = 'Random Forest'
        print("\n--- Configurando Random Forest ---")
        n = input("Número de árboles (ej. 50, 100, 200) [Default 100]: ")
        params['n_estimators'] = int(n) if n.strip() else 100

        d = input("Profundidad máxima (Enter para None, o ej. 10, 20) [Default None]: ")
        params['max_depth'] = int(d) if d.strip() else None

    elif opcion == '3':
        nombre = 'MLP'
        print("\n--- Configurando Red Neuronal (MLP) ---")
        h = input("Capas ocultas (1: (50,), 2: (100,), 3: (50, 50)) [Default 1]: ")
        if h == '2':
            params['hidden_layer_sizes'] = (100,)
        elif h == '3':
            params['hidden_layer_sizes'] = (50, 50)
        else:
            params['hidden_layer_sizes'] = (50,)

        params['max_iter'] = 1000  # Para que no tire warnings de convergencia

    else:
        print("\n[-] Opción no válida. Cargando SVM por defecto.")
        nombre = 'SVM'
        params = {'kernel': 'rbf', 'C': 1.0}

    return nombre, params


def instanciar_modelo(nombre, params):
    if nombre == 'SVM':
        return SVC(**params)
    elif nombre == 'Random Forest':
        return RandomForestClassifier(**params)
    elif nombre == 'MLP':
        return MLPClassifier(**params)
    return None


def iniciar_worker_mlops():
    # El investigador arma su experimento localmente
    nombre, params = obtener_configuracion()

    print(f"\n[*] Conectando al Servidor Maestro en {HOST}:{PORT}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((HOST, PORT))

            # Empaquetar la orden MLOps
            peticion = {
                'accion': 'SOLICITAR_MODELO_MLOPS',
                'modelo': nombre,
                'parametros': params
            }
            enviar_datos(client_socket, peticion)
            print(f"[+] Petición enviada: {nombre} con hiperparámetros: {params}")
            print("[*] Esperando el dataset masivo del servidor...")

            # Recibir las matrices pesadas
            tarea = recibir_datos(client_socket)

            if tarea and tarea.get('accion') == 'ENTRENAR':
                X_train, X_test, y_train, y_test = tarea['datos']
                print(f"\n[+] Dataset recibido. ¡Procesador al 100%! Entrenando {nombre}...")

                start_time = time.time()

                clf = instanciar_modelo(nombre, params)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                t_exec = time.time() - start_time
                print(f"[✓] Entrenamiento finalizado en {t_exec:.2f}s. Accuracy: {acc:.4f}")

                # Devolver resultados para el Leaderboard
                respuesta = {
                    'status': 'EXITO',
                    'resultado': {
                        'modelo': nombre,
                        'params': params,
                        'accuracy': acc,
                        'tiempo': t_exec
                    }
                }
                enviar_datos(client_socket, respuesta)
                print("[*] Resultados enviados al Servidor. Worker desconectado limpiamente.")
            else:
                print("[-] El servidor no devolvió una tarea válida.")

    except ConnectionRefusedError:
        print("[-] Error: El Servidor Maestro está apagado.")
    except Exception as e:
        print(f"[-] Ocurrió un error inesperado: {e}")


if __name__ == '__main__':
    iniciar_worker_mlops()
