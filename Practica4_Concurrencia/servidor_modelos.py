import socket
import threading
from utils import enviar_datos, recibir_datos

HOST = '127.0.0.1'

# RECURSOS COMPARTIDOS (Zona Crítica)
leaderboard = []
candado = threading.Lock()


def manejar_resultado(conn, addr, tipo_servidor):
    with conn:
        peticion = recibir_datos(conn)
        if peticion and peticion.get('accion') == 'REGISTRAR_RESULTADO':
            res = peticion['resultado']

            # --- SINCRONIZACIÓN (MUTEX) ---
            candado.acquire()
            try:
                leaderboard.append(res)
                leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)

                print(f"\n[Hilo-{addr[1]}] [✓] ¡{res['modelo']} registrado!")
                print("-" * 40)
                print(f"🏆 LEADERBOARD {tipo_servidor} 🏆")
                for i, m in enumerate(leaderboard):
                    print(f"{i + 1}. {m['modelo']} | Acc: {m['accuracy']:.4f} | {m['tiempo']:.2f}s")
                print("-" * 40)
            finally:
                candado.release()
            # ------------------------------

            enviar_datos(conn, {'status': 'RECIBIDO'})


def iniciar_tablero():
    print("=== CONFIGURACIÓN DE NODO TABLERO ===")
    print("1. Servidor Matemático (SVM, MLP, Logística) -> Puerto 5001")
    print("2. Servidor de Árboles (RF, Árbol, Naive) -> Puerto 5002")
    opc = input("Elige el rol de este servidor (1-2): ")

    PORT = 5001 if opc == '1' else 5002
    tipo = "MATEMÁTICO" if opc == '1' else "ÁRBOLES"

    print(f"\n--- NODO {tipo} ESCUCHANDO EN {PORT} ---")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        while True:
            conn, addr = server_socket.accept()
            # Concurrencia para recibir resultados al mismo tiempo
            threading.Thread(target=manejar_resultado, args=(conn, addr, tipo)).start()


if __name__ == '__main__':
    iniciar_tablero()
