import socket
import time
import xmlrpc.client
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Seguimos usando utils SOLO para el dataset masivo
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PUERTO_DATOS = 5000
PUERTO_BROKER = 9000


def obtener_modelo():
    print("=== WORKER DISTRIBUIDO (HÍBRIDO: SOCKETS + RMI) ===")
    print("1. SVM\n2. Random Forest\n3. MLP\n4. Árbol de Decisión\n5. Regresión Log.\n6. Naive Bayes")
    opc = input("Elige el modelo a procesar: ")

    if opc == '1':
        return 'SVM', SVC(kernel='linear')
    elif opc == '2':
        return 'Random Forest', RandomForestClassifier(n_estimators=50)
    elif opc == '3':
        return 'MLP', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    elif opc == '4':
        return 'Árbol de Decisión', DecisionTreeClassifier()
    elif opc == '5':
        return 'Regresión Logística', LogisticRegression(max_iter=1000)
    elif opc == '6':
        return 'Naive Bayes', GaussianNB()
    return 'SVM', SVC()


def iniciar_worker():
    nombre, clf = obtener_modelo()

    # --- FASE 1: Consultar al Broker (Magia RMI) ---
    print(f"\n[*] Consultando al Broker RMI en el puerto {PUERTO_BROKER}...")
    broker_remoto = xmlrpc.client.ServerProxy(f"http://{HOST}:{PUERTO_BROKER}/")

    # Invocamos la función como si estuviera en este mismo archivo
    puerto_destino = broker_remoto.obtener_puerto(nombre)

    if puerto_destino == -1:
        print("[-] Error: El Broker no conoce ese modelo.")
        return

    print(f"[+] El Broker asignó el Tablero en el puerto: {puerto_destino}")

    # --- FASE 2: Descargar Datos (Sockets TCP puros) ---
    print(f"\n[*] Conectando al Nodo de Datos (Sockets Puerto {PUERTO_DATOS})...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_datos:
        s_datos.connect((HOST, PUERTO_DATOS))
        enviar_datos(s_datos, {'accion': 'SOLICITAR_DATOS'})
        respuesta = recibir_datos(s_datos)
        X_train, X_test, y_train, y_test = respuesta['datos']
        print("[+] Dataset masivo descargado a máxima velocidad binaria.")

    # --- FASE 3: Procesamiento Local ---
    print(f"\n[*] Entrenando {nombre} (CPU al 100%)...")
    start = time.time()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    t_exec = time.time() - start
    print(f"[✓] Entrenamiento finalizado. Accuracy: {acc:.4f} en {t_exec:.2f}s")

    # --- FASE 4: Reportar al Tablero ---
    print(f"\n[*] Reportando resultados al Tablero RMI (Puerto {puerto_destino})...")
    tablero_remoto = xmlrpc.client.ServerProxy(f"http://{HOST}:{puerto_destino}/")

    # Cero diccionarios, cero pickle manual. Pasamos variables y ya.
    exito = tablero_remoto.registrar_resultado(nombre, float(acc), float(t_exec))

    if exito:
        print("[+] Resultado registrado exitosamente en la red.")


if __name__ == '__main__':
    iniciar_worker()
