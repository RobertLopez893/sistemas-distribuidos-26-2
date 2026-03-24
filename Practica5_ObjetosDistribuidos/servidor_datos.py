import socket
import threading
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'
PORT = 5000


def manejar_cliente(conn, addr, dataset):
    """Hilo independiente para enviar los 100k datos rápido."""
    with conn:
        print(f"[Hilo-{addr[1]}] Worker solicitando datos...")
        peticion = recibir_datos(conn)

        if peticion and peticion.get('accion') == 'SOLICITAR_DATOS':
            enviar_datos(conn, {'status': 'EXITO', 'datos': dataset})
            print(f"[Hilo-{addr[1]}] [✓] Dataset masivo enviado.")


def iniciar_servidor_datos():
    print("--- NODO DE DATOS GLOBALES (PUERTO 5000) ---")
    print("[*] Generando dataset sintético (100,000 muestras)...")
    X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    dataset = train_test_split(X, y, test_size=0.3, random_state=42)
    print("[✓] Dataset listo en RAM. Esperando Workers...\n")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        while True:
            conn, addr = server_socket.accept()
            # Concurrencia pura: un hilo por descarga
            threading.Thread(target=manejar_cliente, args=(conn, addr, dataset)).start()


if __name__ == '__main__':
    iniciar_servidor_datos()