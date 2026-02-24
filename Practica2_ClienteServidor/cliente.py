import socket
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PORT = 5000


def instanciar_modelo(nombre, params):
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
    elif nombre == 'MLP':
        return MLPClassifier(**params, max_iter=1000)
    return None


def iniciar_worker():
    print("=== MENÚ DE WORKER ===")
    print("1. SVM (Support Vector Machine)")
    print("2. Random Forest")
    print("3. Regresión Logística")
    print("4. Árbol de Decisión")
    print("5. Naive Bayes")
    print("6. MLP (Red Neuronal)")

    opcion = input("\nElige el número del modelo que quieres entrenar: ")

    print(f"\n[*] Conectando al Servidor Maestro en {HOST}:{PORT}...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((HOST, PORT))

            # Pedimos el modelo
            peticion = {'accion': 'SOLICITAR_MODELO', 'opcion': opcion}
            enviar_datos(client_socket, peticion)
            print("[*] Petición enviada. Esperando datos...")

            # Recibimos la tarea y los datos
            tarea = recibir_datos(client_socket)

            if tarea and tarea.get('accion') == 'ENTRENAR':
                nombre = tarea['nombre']
                params = tarea['parametros']
                X_train, X_test, y_train, y_test = tarea['datos']

                print(f"\n[+] Datos recibidos. Entrenando {nombre}...")
                start_time = time.time()

                clf = instanciar_modelo(nombre, params)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                end_time = time.time()
                t_exec = end_time - start_time
                print(f"[✓] Entrenamiento finalizado en {t_exec:.2f}s.")

                respuesta = {
                    'status': 'EXITO',
                    'resultado': {'modelo': nombre, 'accuracy': acc, 'tiempo': t_exec}
                }

                # Devolvemos el resultado
                enviar_datos(client_socket, respuesta)
                print("[*] Resultado enviado al servidor.")

    except Exception as e:
        print(f"[-] Error de conexión: {e}")


if __name__ == '__main__':
    iniciar_worker()
