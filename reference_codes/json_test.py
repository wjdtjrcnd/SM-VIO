from MPU6050 import mpu6050  # Ensure you import the class, not the module
import time
import socket
import pickle
import math
import numpy as np
import json

imu = mpu6050(0x68, bus=7)
imu.set_filter_range(0x03)
imu.calibrate()
last = time.time()

try:
    while True:
        sendingData = ''
        
        now = time.time()
        angle = imu.update_angles()
        
        if now - last > 0.1:
            last = now

            # Extract angles in radians
            x = angle['z']  # Rotation around X-axis
            y = -angle['x']   # Rotation around Z-axis
            z = -angle['y'] # Rotation around Y-axis
            
            print(f"X Angle: {math.degrees(x):.3f}, Y Angle: {math.degrees(y):.3f}, Z Angle: {math.degrees(z):.3f}")
            
            array = [x,y,z]
            
            data_str = json.dumps([f"{x:.3f}" for x in array])  # Keep fixed precision
            data_bytes = data_str.encode('utf-8')

            # Send length of data first
            data_length = f"{len(data_bytes):<10}".encode('utf-8')
            simulated_sent_data = data_length + data_bytes
            
            # Simulate receiving by reading the length and data separately
            received_length = int(simulated_sent_data[:10].decode('utf-8').strip())
            received_data_bytes = simulated_sent_data[10:10 + received_length]
            received_data_str = received_data_bytes.decode('utf-8')
            received_array = json.loads(received_data_str)
            received_array = [float(val) for val in received_array]  # Convert back to float

            # Print the encoded data and the "received" data
            print("Encoded data to send:", data_str)
            print("Simulated received angles:", received_array)            
            
            time.sleep(0.005)
            
finally:
    print("IMU terminated")