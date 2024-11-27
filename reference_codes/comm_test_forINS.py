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
        data_list = json.loads(received_data_str)
        data_list = [float(value) for value in data_list]
        
        # Extract Euler angles and position vector
        euler_angles = data_list[:3]  # First three values are Euler angles
        position_vector = data_list[3:]  # Last three values are position vector components
        
        print("Received Euler angles:", euler_angles)
        print("Received position vector:", position_vector)
        
finally:
    conn.close()
    server_socket.close()
    print("Server socket closed.")
