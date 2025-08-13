import socket
import threading
import zlib
import json
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
import os

class Server:
    def __init__(self, host='0.0.0.0', port=65432, use_ipv6=False):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET6 if use_ipv6 else socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.dh_params = dh.generate_parameters(generator=2, key_size=2048)

    def process_request(self, data):
        try:
            request = json.loads(data.decode())
            if request['type'] == 'ask':
                # Пример обработки текстового запроса
                return json.dumps({"result": f"Ответ: {request['data']}"}).encode()
            elif request['type'] == 'emb':
                # Возвращаем эмбеддинги (без сжатия)
                return json.dumps({"embedding": [0.1 * i for i in range(128)]}).encode()
        except:
            return b"ERROR: INVALID_REQUEST"

    def handle_client(self, conn):
        try:
            # 1. Отправляем параметры DH
            conn.sendall(self.dh_params.parameter_bytes(serialization.Encoding.PEM, serialization.ParameterFormat.PKCS3))

            while True:
                # 2. Получаем публичный ключ клиента или проверочный запрос
                client_pub_pem = conn.recv(4096)
                if not client_pub_pem:
                    # Это проверочный запрос - возвращаем 122
                    conn.sendall(b'122')
                    break

                # 3. Генерируем временный ключ сервера
                server_priv = self.dh_params.generate_private_key()
                conn.sendall(server_priv.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))

                # 4. Вычисляем общий секрет
                client_pub = serialization.load_pem_public_key(client_pub_pem)
                shared_secret = server_priv.exchange(client_pub)
                aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'aes_key').derive(shared_secret)

                # 5. Получаем и расшифровываем сообщение
                encrypted = conn.recv(4096)
                iv, tag, ciphertext = encrypted[:16], encrypted[-16:], encrypted[16:-16]
                cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag), backend=default_backend())
                decrypted = cipher.decryptor().update(ciphertext) + cipher.decryptor().finalize()

                # 6. Обработка запроса (сжатие только для текста)
                response = self.process_request(decrypted)
                if b'"result":' in response:  # Текстовый ответ
                    response = zlib.compress(response)

                # 7. Шифруем ответ
                iv = os.urandom(16)
                cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv), backend=default_backend())
                encrypted_resp = iv + cipher.encryptor().update(response) + cipher.encryptor().finalize() + cipher.encryptor().tag
                conn.sendall(encrypted_resp)

        finally:
            conn.close()

    def start(self):
        self.sock.listen()
        print(f"Сервер слушает {self.host}:{self.port} (IPv{'6' if ':' in self.host else '4'})")
        while True:
            conn, addr = self.sock.accept()
            threading.Thread(target=self.handle_client, args=(conn,)).start()

if __name__ == "__main__":
    server = Server()  # Для IPv6: Server(host='::', use_ipv6=True)
    server.start()