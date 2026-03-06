import socket
import threading
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Importamos nuestras herramientas de red
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PORT = 5000

leaderboard_global = []  # Aquí guardaremos todos los resultados
candado_leaderboard = threading.Lock()  # El Mutex para proteger la lista


def manejar_cliente(conn, addr, dataset):
    """
    Función que ejecuta CADA HILO de forma independiente.
    Atiende a un solo cliente de principio a fin.
    """
    X_train, X_test, y_train, y_test = dataset

    with conn:
        print(f"\n[Hilo-{addr[1]}] [+] Worker conectado desde {addr}")

        try:
            # Escuchar qué experimento MLOps quiere hacer el cliente
            peticion = recibir_datos(conn)

            if peticion and peticion.get('accion') == 'SOLICITAR_MODELO_MLOPS':
                nombre = peticion['modelo']
                params = peticion['parametros']
                print(f"[Hilo-{addr[1]}] [*] El Worker solicitó entrenar: {nombre} con {params}")

                # Le mandamos el dataset completo
                print(f"[Hilo-{addr[1]}] [*] Enviando dataset pesado al Worker...")
                tarea = {'accion': 'ENTRENAR', 'datos': (X_train, X_test, y_train, y_test)}
                enviar_datos(conn, tarea)

                # Esperamos el resultado (Este hilo se bloquea, ¡pero los demás siguen vivos!)
                print(f"[Hilo-{addr[1]}] [*] Esperando resultados de {nombre}...")
                respuesta = recibir_datos(conn)

                if respuesta and respuesta.get('status') == 'EXITO':
                    res = respuesta['resultado']

                    # Pedimos la llave. Si otro hilo la tiene, este se espera.
                    candado_leaderboard.acquire()
                    try:
                        # Ya tenemos la llave, es seguro modificar la lista global
                        leaderboard_global.append(res)

                        # Ordenamos el leaderboard para ver quién va ganando
                        leaderboard_global.sort(key=lambda x: x['accuracy'], reverse=True)

                        print(f"\n[Hilo-{addr[1]}] [✓] ¡Resultado recibido de {nombre}! (Acc: {res['accuracy']:.4f})")
                        print("-" * 50)
                        print("🏆 LEADERBOARD GLOBAL ACTUALIZADO 🏆")
                        for i, modelo in enumerate(leaderboard_global):
                            print(
                                f"{i + 1}. {modelo['modelo']} | Acc: {modelo['accuracy']:.4f} | Tiempo: {modelo['tiempo']:.2f}s | Params: {modelo['params']}")
                        print("-" * 50)
                    finally:
                        # Soltar la llave pase lo que pase
                        candado_leaderboard.release()

        except Exception as e:
            print(f"\n[Hilo-{addr[1]}] [-] Error con el Worker: {e}")


def iniciar_servidor_concurrente():
    print("--- INICIO: Generando Dataset Sintético (100,000 muestras) ---")
    X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    dataset = train_test_split(X, y, test_size=0.3, random_state=42)
    print("[✓] Dataset listo en memoria.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"\n[*] Servidor Maestro Concurrente escuchando en {HOST}:{PORT}...")

        while True:
            # EL HILO PRINCIPAL
            conn, addr = server_socket.accept()

            # Le pasamos la conexión, la IP y una copia de referencia del dataset
            hilo_worker = threading.Thread(target=manejar_cliente, args=(conn, addr, dataset))

            # Iniciamos el hilo y el portero vuelve inmediatamente al accept()
            hilo_worker.start()
            print(f"[*] Hilo despachador creado para {addr}. (Hilos activos: {threading.active_count() - 1})")


if __name__ == '__main__':
    iniciar_servidor_concurrente()
