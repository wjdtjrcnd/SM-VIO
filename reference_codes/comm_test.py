import socket
import json

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 12345))
server_socket.listen(1)
print("Server is listening on port 12345...")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

try:
    while True:
        # Receive the length of the incoming data
        data_length = int(conn.recv(10).decode('utf-8').strip())
        
        # Now receive the actual data based on the received length
        data_bytes = conn.recv(data_length)
        received_data_str = data_bytes.decode('utf-8')

        # Convert JSON string to list of floats
        angles = json.loads(received_data_str)
        angles = [float(angle) for angle in angles]
        print("Received angles:", angles)
        
finally:
    conn.close()
    server_socket.close()
    print("Server socket closed.")
