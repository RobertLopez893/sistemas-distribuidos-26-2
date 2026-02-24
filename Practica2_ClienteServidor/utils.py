import pickle
import struct


def enviar_datos(sock, datos):
    """Convierte un objeto de Python a bytes y lo envía por el socket."""
    # Serializar el objeto (ej. diccionario, arreglo de numpy) a bytes
    datos_bytes = pickle.dumps(datos)

    # Empaquetar el tamaño del mensaje en los primeros 4 bytes (Formato '>I')
    mensaje_con_tamano = struct.pack('>I', len(datos_bytes)) + datos_bytes

    # Enviar por la red
    sock.sendall(mensaje_con_tamano)


def recibir_datos(sock):
    """Lee el socket asegurándose de recibir el mensaje completo y lo deserializa."""
    # Leer los primeros 4 bytes para saber el tamaño del paquete
    raw_msglen = leer_n_bytes(sock, 4)
    if not raw_msglen:
        return None

    # Desempaquetar esos 4 bytes para obtener un número entero
    msglen = struct.unpack('>I', raw_msglen)[0]

    # Leer exactamente esa cantidad de bytes del socket
    datos_bytes = leer_n_bytes(sock, msglen)

    # Deserializar los bytes de vuelta a un objeto de Python
    return pickle.loads(datos_bytes)


def leer_n_bytes(sock, n):
    """Función auxiliar para asegurar que leemos exactamente 'n' bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
