import os
import Jetson.GPIO as GPIO
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
from visual_odometry import VisualOdometry, PinholeCamera
import torch
import torchvision
import cv2
import time
import numpy as np
import PIL.Image
import socket
import pickle
from math import sqrt
from Kalman import KalmanFilter
from cnn.center_dataset import TEST_TRANSFORMS
from jetcam.csi_camera import CSICamera
import glob
from depth import DepthEngine
'''
Run mpu6050test_with_vo_0613.py with this code

If Gstreamer Related Error Occurs
execute this in terminal:
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0:/lib/aarch64-linux-gnu/libpthread.so.0
'''

def get_lane_model():
        lane_model = torchvision.models.alexnet(num_classes=2, dropout=0.3)
        return lane_model

def preprocess(image: PIL.Image):
        device = torch.device('cuda')    
        image = TEST_TRANSFORMS(image).to(device)
        return image[None, ...]

def run_odom(frame, img_id, array):
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # receive data from IMU
    #array = pickle.loads(data, encoding='utf-8')
    acc = np.zeros((3,1))
    if data is not None:
        acc = np.array(array[0])
        acc = acc.reshape((3,1))
        #a = data.split()
        #acc[0] = float(a[0].strip().strip(',[]'))
        #acc[1] = -float(a[2].strip().strip(',[]'))
        #acc[2] = -float(a[1].strip().strip(',[]'))

        vo.update(grayimg, img_id, array)
        
        cur_t = vo.cur_t
        if(img_id > 0):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
            kf.predict(acc)
            kf.update(cur_t.reshape(3,1))
            x,y,z = kf.get_state()
        else:
            x, y, z = 0., 0., 0.
    # draw_x, draw_y = -int(x)+290, int(z)+90
    # cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
    # cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    # text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    # cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    return x,y,z

def rot2eul(Rot):
    beta = np.arcsin(-Rot[2,0]) # theta (pitch)
    alpha = np.arctan2(Rot[2,1], Rot[2,2]) # phi (yaw)
    gamma = np.arctan2(Rot[1,0], Rot[0,0]) # psi (roll)
    return np.array((alpha, beta, gamma))


camera = CSICamera(capture_width=960, capture_height=540, downsample = 1, capture_fps=30)

car = NvidiaRacecar()
car.steering_gain = -1.0
car.steering_offset = 0.2                                                       #do not change
car.throttle_gain = 0.5
steering_range = (-1.0 + car.steering_offset, 1.0 + car.steering_offset)
car.throttle = 0.0
car.steering = 0.0

#GPIO Pin Setup
LED_R = 35
LED_G = 33
LED_B = 31
Vcc   = 37
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(Vcc, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(LED_R, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(LED_G, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(LED_B, GPIO.OUT, initial=GPIO.HIGH)

#Lane tracking model setup
device = torch.device('cuda')
lane_model = get_lane_model()
lane_model.load_state_dict(torch.load('road_following_model_alexnet_best_0.3_lbatch.pth'))
lane_model = lane_model.to(device)

#Visual Odometry setup

cam = PinholeCamera(960.0, 540.0, 636.65410484, 635.66094765, 489.56491003, 266.9754490)
vo = VisualOdometry(cam)

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

#set the  values for throttle, steering and control 
Kp_throttle = 0.053
Kp_throttle_flat = 0.048
target_displacement = 1.0
Kp = 2.0
Kd = 0.08
Ki = 0.3
#Kp = 1.0
#Ki = 3.0
cruisethrottle = 0.2450
maxthrottle = 0.42
minthrottle = 0.0
brakethrottle = -0.2
throttle_increment = 0.006
kf = KalmanFilter(0.2, 0.01, 0.3)

turn_threshold = 0.75
integral_threshold = 0.2
integral_range = (-0.4/Ki, 0.4/Ki)                                              #integral threshold for saturation prevention

#odometry initialization
img_id = 0
#dir = glob.glob('*.txt')
#last = dir[-1]
#num = int(last[8:last.find(".")])
#store_dir = "./log/odometry" + str(num+1) +".txt"
#with open(store_dir, "w+") as f:                                           #txt file for recording odometry
#    f.truncate(0)
#f = open(store_dir, 'w')
# traj = np.zeros((600,600,3), dtype=np.uint8)
with open("odometry4.txt", "w+") as f:                                           #txt file for recording odometry
    f.truncate(0)
f = open("odometry4.txt", 'w')

dir = glob.glob('1.txt')
#dir[-1] - '.txt' + '_' + dir[-1][-4]
with open("./log/1.txt", "w+") as f2:                                           #txt file for recording accel
    f2.truncate(0)
f2 = open("./log/1.txt", 'w')

#initializing values
execution = True
inclination_flag = False
exit_flag = False
boost_flag = False
brake_flag = False
brake_ignore = False
YOLO_flag = True
previous_err = 0.
integral = 0.
x_prev = 0.
y_prev = 0.
z_prev = 0.

# Load the Depth model
depth = DepthEngine(
        frame_rate=10,
        raw=True,
    )

try:
    #print(car.throttle)
    throttle = cruisethrottle
    now = time.time()
    # socket communication with IMU program
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '127.0.0.1'
    port = 12345
    server_socket.bind((host, port))
    server_socket.listen(5)
    print("에코 서버가 시작되었습니다.")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"클라이언트가 연결되었습니다: {addr}")
        if client_socket is not None:
            break
        
    print("Ready...")
    
    GPIO.output(LED_R, GPIO.LOW)
    overall_prev = time.time()
    initiationtime = overall_prev
    obstacleflag = False
    
    while execution:
        obstacleflag = False

        '''
        Receive the data from client socket
        '''
        data = client_socket.recv(297)
        array = pickle.loads(data)
        
        # write the acc for the client socket
        acc = array[0]
        f2.write(f"{acc[0]} {acc[1]} {acc[2]}\n")

        '''
        Jostick Control
        '''
        if joystick.get_button(11): #for shutoff: press start button
            print("stopping...")
            execution = False

        if joystick.get_button(0):
            k = 0.001
            cruisethrottle += k
            # maxthrottle += k
            print("Throttle : ", cruisethrottle)
        
        elif joystick.get_button(1):
            k = -0.001
            cruisethrottle += k
            # maxthrottle += k
            print("Throttle : ", cruisethrottle)

        # Read the image of csi cam
        image = camera.read()

        # Run Visual Odometry
        x,y,z = run_odom(frame=image, img_id=img_id, array=array)
        f.write(f"{z} {-x} {-y}\n")
        img_id += 1
        if abs(z) > 30.0 : YOLO_flag = False
        # Run lane tracking
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_coverted)
        pil_width = pil_image.width
        pil_height = pil_image.height
        with torch.no_grad():
            image2 = preprocess(image=pil_image)
            output = lane_model(image2).detach().cpu().numpy()
        err, y = output[0]

        '''
        Steering Control
        '''
        # Calculate time interval
        time_interval = time.time()-now
        now = time.time()
        
        #Anti-windup
        if abs(err)> integral_threshold: integral = 0                                   #prevent output saturation
        elif previous_err * err< 0: integral = 0                                        #zero-crossing reset
        else:
            integral += err * time_interval
            integral = max(integral_range[0], min(integral_range[1], integral))         #prevent integral saturation
        steering = float(Kp*err+Kd*(err-previous_err)/time_interval + Ki*integral)
        steering = max(steering_range[0], min(steering_range[1], steering))
        previous_err = err
        
        '''
        Find the odom
        '''
        # detect inclination        
        ROT = np.array(array[1])
        roll, pitch, yaw = rot2eul(ROT)
        
        # Find the displacement
        displacement = round(sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2),4)
        x_prev = x
        y_prev = y
        z_prev = z
        dis_err = target_displacement - displacement
        
        #LED control
        GPIO.output(LED_G,GPIO.HIGH)
        GPIO.output(LED_R, GPIO.HIGH)
        GPIO.output(LED_B, GPIO.HIGH)
        
        if  roll < -0.15 :
            inclination_flag = True
            GPIO.output(LED_B, GPIO.LOW)
            GPIO.output(LED_R, GPIO.HIGH)
            GPIO.output(LED_G, GPIO.HIGH)
                     
        elif displacement > 0.15:
            GPIO.output(LED_R,GPIO.HIGH)
            GPIO.output(LED_G, GPIO.LOW)
            GPIO.output(LED_B, GPIO.HIGH)
            
        elif abs(roll) < 0.15 and displacement < 0.15:
            GPIO.output(LED_G,GPIO.HIGH)
            GPIO.output(LED_R, GPIO.LOW)
            GPIO.output(LED_B, GPIO.HIGH)
        
        
            
        
        overall_now = time.time()
        
        '''
        Print the Result
        '''
        os.system('clear')
        print("-----------------------------")
        print("connected with client")
        print("x,y,z : " , z, -x, -y)
        print("displacement = ",displacement)
        print("Euler",round(roll,5),round(pitch,5),round(yaw,5))
        # print("ROT:", ROT)
        print("acc", acc[0],acc[1],acc[2])
        print("time interval", overall_now-overall_prev)
        print("throttle:", throttle)
        print("err : ",err)
        overall_prev = overall_now
        
        #image = cv2.resize(image, (480, 270))
        if YOLO_flag:
            depth_data = depth.infer(image,1)
            #flag = False
            if depth_data is not None:
                #print(depth_data.shape)
                #height, width = depth_data.shape
                mean_value_axis1 = np.mean(depth_data[100:360,260:700]) 
                print("mean :" , mean_value_axis1)
                print("mean2 : ", np.mean(depth_data[100:360,320:640]))
                #max_values = np.max(depth_data, axis=1)
                cnt = 0
                #print(depth_data.shape)
                #print(np.mean(depth_data))
                #print(np.mean(depth_data[100:360][320:640]))
                    
                for i in range(100, 360, 5):
                    for j in range(320, 640, 5):
                        if depth_data[i][j] > mean_value_axis1 * 1.15:
                            cnt += 1
                print("cnt: " ,cnt)
                if cnt > 1400:
                    obstacleflag = True
        
        
        '''
        Throttle Control
        '''
        if roll > 0.06 and not brake_ignore:
            throttle = brakethrottle
            print("emrgency braking")
            if not brake_flag:
                braketime = now
                brake_flag = True
            if (now - braketime) > 2.0: brake_ignore = True

        elif inclination_flag and roll > -0.05:                                                 #trying to stop on top of hill
            if not exit_flag:
                exit_flag = True
                exit_time = now
            if now - exit_time > 2.0:
                inclination_flag = False
                exit_flag = False
            throttle = brakethrottle
             
        # elif displacement > 1.2:
        #     throttle = max(minthrottle, (throttle + Kp_throttle*dis_err))
        #     print("decelerating")
        
        # elif ((now - initiationtime) > 10.0 and displacement < 0.75):
        #     if not boost_flag:
        #         print("boosting...")
        #         boost_flag = True
        #         boost_time = now
        #     if displacement < 0.1:
                
        #         throttle = min(maxthrottle, (throttle + Kp_throttle*dis_err))
        
        elif displacement < 0.2 and roll > -0.1 and (now-initiationtime) > 3:
            throttle = min(maxthrottle, (throttle + throttle_increment))
            print("accelerating...")
        
        elif displacement < 0.2 and roll < -0.1 and (now-initiationtime) > 3:
            throttle = min(maxthrottle, (throttle + throttle_increment))
            print("accelerting")
            
        elif displacement > 1.0 :
            throttle = max(minthrottle, (throttle - throttle_increment))
            print("deccelerating...")
        
        # elif abs(steering-0.2) > turn_threshold:
        #     throttle = cruisethrottle + 0.025

        else: throttle = cruisethrottle

        if obstacleflag and abs(roll) < 0.2:
            print("STOP!!!")
            throttle = 0.0
            
        car.steering = steering
        car.throttle = throttle
        pygame.event.pump()
        

finally:
    f.close()
    f2.close()
    # cv2.imwrite('map.png', traj)
    camera.release()
    print("odometry saved... program terminated")
    car.throttle = 0.0
    car.steering = 0.0
    GPIO.output(LED_R, GPIO.LOW)
    GPIO.output(LED_G, GPIO.LOW)
    GPIO.output(LED_B, GPIO.LOW)
    GPIO.output(Vcc,   GPIO.LOW)