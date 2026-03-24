from xmlrpc.server import SimpleXMLRPCServer
import threading


class TableroEspecialista:
    """Este es nuestro Objeto Distribuido con Estado y Comportamiento"""

    def __init__(self, tipo_servidor):
        # El ESTADO del objeto
        self.leaderboard = []
        self.candado = threading.Lock()  # ¡La sincronización sigue intacta!
        self.tipo_servidor = tipo_servidor

    # EL COMPORTAMIENTO (Método Remoto)
    def registrar_resultado(self, modelo, accuracy, tiempo):
        """Este método será ejecutado mágicamente por los clientes a través de la red"""

        # --- ZONA CRÍTICA: PROTEGIDA POR EL MUTEX ---
        with self.candado:
            # Guardamos el nuevo récord
            nuevo_record = {'modelo': modelo, 'accuracy': accuracy, 'tiempo': tiempo}
            self.leaderboard.append(nuevo_record)

            # Ordenamos el tablero de mayor a menor accuracy
            self.leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)

            # Imprimimos el Leaderboard localmente
            print(f"\n[✓] ¡{modelo} registrado por Invocación Remota!")
            print("-" * 40)
            print(f"🏆 LEADERBOARD {self.tipo_servidor} 🏆")
            for i, m in enumerate(self.leaderboard):
                print(f"{i + 1}. {m['modelo']} | Acc: {m['accuracy']:.4f} | {m['tiempo']:.2f}s")
            print("-" * 40)

        # El return viaja automáticamente por la red de regreso al cliente
        return True


def iniciar_tablero():
    print("=== CONFIGURACIÓN DE NODO TABLERO (OBJETO DISTRIBUIDO) ===")
    print("1. Servidor Matemático (SVM, MLP, Logística) -> Puerto 5001")
    print("2. Servidor de Árboles (RF, Árbol, Naive) -> Puerto 5002")
    opc = input("Elige el rol de este servidor (1-2): ")

    PORT = 5001 if opc == '1' else 5002
    tipo = "MATEMÁTICO" if opc == '1' else "ÁRBOLES"

    # PASO 1: Instanciamos nuestro objeto
    mi_objeto_tablero = TableroEspecialista(tipo)

    # PASO 2: Creamos el Servidor RMI
    servidor_rmi = SimpleXMLRPCServer(('127.0.0.1', PORT), allow_none=True)

    # PASO 3: Publicamos el objeto en la red
    servidor_rmi.register_instance(mi_objeto_tablero)

    print(f"\n--- NODO {tipo} (RMI) ESCUCHANDO EN EL PUERTO {PORT} ---")

    # Lo dejamos corriendo
    servidor_rmi.serve_forever()


if __name__ == '__main__':
    iniciar_tablero()
