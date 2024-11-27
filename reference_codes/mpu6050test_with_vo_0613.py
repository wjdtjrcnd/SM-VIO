from jet_imu.MPU6050custom import mpu6050  # Ensure you import the class, not the module
import time
import socket
import pickle
import math
import json
import os

imu = mpu6050()
imu.set_filter_range(0x03)
imu.calibrate()
last = time.time()

# 서버 소켓 생성
#server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버 소켓을 특정 포트에 바인딩
server_host = '127.0.0.1'
server_port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_host, server_port))

with open("./log/2.txt", "w+") as f2:                                           #txt file for recording odometry
    f2.truncate(0)
f2 = open("./log/2.txt", 'w')

#time.sleep(4)
try: 
    i = 0
    while True:
        
        sendingData = ''
        now = time.time()
        data = imu.update()
        if now - last > 0.1:
            i = 1
            last = now
            ROT = data['rot']
            speed = data['v']
            velocity = math.sqrt(speed[0]**2 + speed[1]**2 + speed[2]**2)
            acc = [data['ax'],data['ay'],data['az']]
            array = [acc,ROT]
            
            f2.write(f"{acc[0]} {acc[1]} {acc[2]}\n")
            #sendingData = json.dumps(array)
            sendingData = pickle.dumps(array)
            os.system('clear')
            print("---------------------")
            print("connected with server")
            print("SendingData : \n", array)
            print("SendingData's length : ", len(sendingData))
            #sendingData = str(sendingData)
            
            client_socket.sendall(sendingData)
            
            #below is for sending Rotation matrix and velocity
            '''
            # Serialize the data
            serialized_data = pickle.dumps({'ROT': ROT, 'velocity': velocity})
            # Send the length of the serialized data
            client_socket.sendall(len(serialized_data).to_bytes(4, byteorder='big'))

            # Send the actual serialized data
            client_socket.sendall(serialized_data)
            '''
        time.sleep(0.005)
            
finally:
    f2.close()