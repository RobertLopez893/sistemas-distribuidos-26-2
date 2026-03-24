import socket
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PUERTO_DATOS = 5000

# El Cliente sabe a qué puerta tocar para reportar su modelo
RUTAS = {
    'SVM': 5001,
    'MLP': 5001,
    'Regresión Logística': 5001,
    'Random Forest': 5002,
    'Árbol de Decisión': 5002,
    'Naive Bayes': 5002
}


def obtener_modelo():
    print("=== WORKER DISTRIBUIDO ===")
    print("1. SVM\n2. Random Forest\n3. MLP\n4. Árbol de Decisión\n5. Regresión Log.\n6. Naive Bayes")
    opc = input("Elige el modelo a procesar: ")

    if opc == '1':
        return 'SVM', SVC(kernel='linear')  # Simplificado para rapidez
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
    puerto_destino = RUTAS[nombre]

    # PASO 1: Ir al Nodo de Datos (5000)
    print(f"\n[*] Conectando al Nodo de Datos (Puerto {PUERTO_DATOS})...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_datos:
        s_datos.connect((HOST, PUERTO_DATOS))
        enviar_datos(s_datos, {'accion': 'SOLICITAR_DATOS'})
        respuesta = recibir_datos(s_datos)
        X_train, X_test, y_train, y_test = respuesta['datos']
        print("[+] Dataset masivo descargado correctamente.")

    # PASO 2: Procesamiento Local
    print(f"[*] Entrenando {nombre} (CPU al 100%)...")
    start = time.time()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    t_exec = time.time() - start
    print(f"[✓] Entrenamiento finalizado. Accuracy: {acc:.4f} en {t_exec:.2f}s")

    # PASO 3: Ir al Nodo Tablero correspondiente (5001 o 5002)
    print(f"[*] Reportando resultados al Servidor Especialista (Puerto {puerto_destino})...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_tablero:
        s_tablero.connect((HOST, puerto_destino))
        payload = {
            'accion': 'REGISTRAR_RESULTADO',
            'resultado': {'modelo': nombre, 'accuracy': acc, 'tiempo': t_exec}
        }
        enviar_datos(s_tablero, payload)
        print("[+] Resultado registrado exitosamente en la red.")


if __name__ == '__main__':
    iniciar_worker()
