import socket
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PORT = 5000


def iniciar_servidor_ml():
    print("--- INICIO: Generando Dataset Sintético (10,000 muestras para que pese) ---")
    # Usamos 10,000 para que el SVM tarde lo suficiente y se note el bloqueo
    X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # El Menú de modelos disponibles
    catalogo_modelos = {
        '1': ('SVM', {'kernel': 'rbf'}),
        '2': ('Random Forest', {'n_estimators': 100}),
        '3': ('Regresión Logística', {'C': 1.0}),
        '4': ('Árbol de Decisión', {'criterion': 'gini'}),
        '5': ('Naive Bayes', {}),
        '6': ('MLP', {'hidden_layer_sizes': (50,)})
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"\n[*] Servidor Maestro escuchando en {HOST}:{PORT}...")

        while True:
            print("\n[*] Esperando peticiones de los Workers...")
            conn, addr = server_socket.accept()

            with conn:
                print(f"[+] Worker conectado desde {addr}")

                # Escuchamos qué quiere el cliente
                peticion = recibir_datos(conn)

                if peticion and peticion.get('accion') == 'SOLICITAR_MODELO':
                    opcion = peticion['opcion']

                    if opcion in catalogo_modelos:
                        nombre, params = catalogo_modelos[opcion]
                        print(f"[*] El Worker solicitó entrenar: {nombre}")

                        paquete_tarea = {
                            'accion': 'ENTRENAR',
                            'nombre': nombre,
                            'parametros': params,
                            'datos': (X_train, X_test, y_train, y_test)
                        }

                        # Le mandamos los datos
                        print(f"[*] Enviando dataset para {nombre}...")
                        enviar_datos(conn, paquete_tarea)

                        # Esperamos el resultado (AQUÍ OCURRE EL CUELLO DE BOTELLA)
                        print(f"[*] Esperando resultados de {nombre}... (Servidor Ocupado)")
                        respuesta = recibir_datos(conn)

                        if respuesta and respuesta.get('status') == 'EXITO':
                            res = respuesta['resultado']
                            print(
                                f"[✓] ¡Recibido! {nombre} -> Accuracy: {res['accuracy']:.4f} (Tiempo: {res['tiempo']:.2f}s)")
                    else:
                        print("[-] Opción no válida solicitada por el Worker.")


if __name__ == '__main__':
    iniciar_servidor_ml()
