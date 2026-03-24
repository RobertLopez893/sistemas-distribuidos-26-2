from xmlrpc.server import SimpleXMLRPCServer
import logging

# Configuramos un poco de logging para ver qué pasa en la consola
logging.basicConfig(level=logging.INFO)

# --- EL ESTADO DEL OBJETO (El Directorio) ---
# El Broker sabe en qué puerto vive cada especialista
DIRECTORIO_RUTAS = {
    'SVM': 5001,
    'Regresión Logística': 5001,
    'MLP': 5001,
    'Random Forest': 5002,
    'Árbol de Decisión': 5002,
    'Naive Bayes': 5002
}


# --- EL COMPORTAMIENTO DEL OBJETO (El Método Remoto) ---
def obtener_puerto_para(modelo):
    """
    Este método será invocado remotamente por los clientes.
    Recibe el nombre del modelo y devuelve el puerto correspondiente.
    """
    puerto = DIRECTORIO_RUTAS.get(modelo)
    if puerto:
        logging.info(f"[Broker] Cliente preguntó por '{modelo}'. Redirigiendo al puerto {puerto}...")
        return puerto
    else:
        logging.warning(f"[Broker] Cliente preguntó por '{modelo}', pero no está en el directorio.")
        return -1  # Retornamos -1 como código de error si el modelo no existe


# --- INICIALIZACIÓN DEL SERVIDOR RMI ---
def iniciar_broker():
    HOST = '127.0.0.1'
    PORT = 9000

    # Creamos el servidor RMI
    servidor_rmi = SimpleXMLRPCServer((HOST, PORT), allow_none=True)

    # Registramos la función para que sea accesible por la red
    servidor_rmi.register_function(obtener_puerto_para, "obtener_puerto")

    print(f"=== BROKER RMI (DIRECTORIO) INICIADO ===")
    print(f"[*] Escuchando invocaciones remotas en {HOST}:{PORT}...\n")

    # Dejamos al servidor corriendo para siempre
    servidor_rmi.serve_forever()


if __name__ == '__main__':
    iniciar_broker()
