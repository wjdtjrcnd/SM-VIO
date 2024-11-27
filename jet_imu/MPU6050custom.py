import smbus
import time
import math
import numpy as np

class mpu6050:
    # Constants
    GRAVITIY_MS2 = 9.80665
    RAD_TO_DEG = 57.2957795131
    DEG_TO_RAD = 0.0174532925199
    ACCEL_TRANSFORMATION_NUMBER = 0.00006103515
    GYRO_TRANSFORMATION_NUMBER = 0.01525878906
    DEFAULT_ACCEL_COEFF = 0.10
    DEFAULT_GYRO_COEFF = 1 - DEFAULT_ACCEL_COEFF

    DISCARDED_MEASURES = 100
    CALIBRATION_MEASURES = 2000

    # Scale Modifiers
    ACCEL_SCALE_MODIFIER_2G = 16384.0
    ACCEL_SCALE_MODIFIER_4G = 8192.0
    ACCEL_SCALE_MODIFIER_8G = 4096.0
    ACCEL_SCALE_MODIFIER_16G = 2048.0

    GYRO_SCALE_MODIFIER_250DEG = 131.0
    GYRO_SCALE_MODIFIER_500DEG = 65.5
    GYRO_SCALE_MODIFIER_1000DEG = 32.8
    GYRO_SCALE_MODIFIER_2000DEG = 16.4

    # Pre-defined ranges
    ACCEL_RANGE_2G = 0x00
    ACCEL_RANGE_4G = 0x08
    ACCEL_RANGE_8G = 0x10
    ACCEL_RANGE_16G = 0x18

    GYRO_RANGE_250DEG = 0x00
    GYRO_RANGE_500DEG = 0x08
    GYRO_RANGE_1000DEG = 0x10
    GYRO_RANGE_2000DEG = 0x18

    FILTER_BW_256 = 0x00
    FILTER_BW_188 = 0x01
    FILTER_BW_98 = 0x02
    FILTER_BW_42 = 0x03
    FILTER_BW_20 = 0x04
    FILTER_BW_10 = 0x05
    FILTER_BW_5 = 0x06

    # MPU-6050 Registers
    PWR_MGMT_1 = 0x6B
    PWR_MGMT_2 = 0x6C

    ACCEL_XOUT0 = 0x3B
    ACCEL_YOUT0 = 0x3D
    ACCEL_ZOUT0 = 0x3F

    TEMP_OUT0 = 0x41

    GYRO_XOUT0 = 0x43
    GYRO_YOUT0 = 0x45
    GYRO_ZOUT0 = 0x47

    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG = 0x1B
    MPU_CONFIG = 0x1A

    def __init__(self, address=0x68, bus=1):
        self.address = address
        self.bus = smbus.SMBus(bus)
        # Wake up the MPU-6050 since it starts in sleep mode
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
        self.filterAccelCoeff = self.DEFAULT_ACCEL_COEFF
        self.filterGyroCoeff = self.DEFAULT_GYRO_COEFF
        self.gyroXoffset = 0
        self.gyroYoffset = 0
        self.gyroZoffset = 0
        self.accXoffset = 0
        self.accYoffset = 0
        self.accZoffset = 0
        self.angGyroX = 0
        self.angGyroY = 0
        self.angGyroZ = 0
        self.angX = 0.0
        self.angY = 0.0
        self.angZ = 0.0
        self.speed = np.zeros(3,dtype=float)
        self.ROT = np.eye(3, dtype=float)
        self.conv = np.array([[1,0,0],                           #rotation matrix for converting to vehicle coordin.
                              [0,0,-1],
                              [0,1,0]])
        self.accThreshold = 0.3
        self.intervalStart = time.time()

    def read_i2c_word(self, register):
        """Read two i2c registers and combine them."""
        high = self.bus.read_byte_data(self.address, register)
        low = self.bus.read_byte_data(self.address, register + 1)
        value = (high << 8) + low
        if value >= 0x8000:
            return -((65535 - value) + 1)
        else:
            return value

    def get_temp(self):
        """Reads the temperature from the onboard temperature sensor of the MPU-6050."""
        raw_temp = self.read_i2c_word(self.TEMP_OUT0)
        actual_temp = (raw_temp / 340.0) + 36.53
        return actual_temp

    def set_accel_range(self, accel_range):
        """Sets the range of the accelerometer."""
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, accel_range)

    def read_accel_range(self, raw=False):
        """Reads the range the accelerometer is set to."""
        raw_data = self.bus.read_byte_data(self.address, self.ACCEL_CONFIG)
        if raw:
            return raw_data
        else:
            if raw_data == self.ACCEL_RANGE_2G:
                return 2
            elif raw_data == self.ACCEL_RANGE_4G:
                return 4
            elif raw_data == self.ACCEL_RANGE_8G:
                return 8
            elif raw_data == self.ACCEL_RANGE_16G:
                return 16
            else:
                return -1

    def get_accel_data(self, g=False):
        """Gets and returns the X, Y and Z values from the accelerometer."""
        x = self.read_i2c_word(self.ACCEL_XOUT0)
        y = self.read_i2c_word(self.ACCEL_YOUT0)
        z = self.read_i2c_word(self.ACCEL_ZOUT0)
        accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G
        accel_range = self.read_accel_range(True)
        if accel_range == self.ACCEL_RANGE_2G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G
        elif accel_range == self.ACCEL_RANGE_4G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_4G
        elif accel_range == self.ACCEL_RANGE_8G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_8G
        elif accel_range == self.ACCEL_RANGE_16G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_16G
        x = x / accel_scale_modifier
        y = y / accel_scale_modifier
        z = z / accel_scale_modifier
        if g:
            return {'x': x, 'y': y, 'z': z}
        else:
            x = x * self.GRAVITIY_MS2
            y = y * self.GRAVITIY_MS2
            z = z * self.GRAVITIY_MS2
            return {'x': x, 'y': y, 'z': z}

    def set_gyro_range(self, gyro_range):
        """Sets the range of the gyroscope."""
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, gyro_range)

    def set_filter_range(self, filter_range=FILTER_BW_256):
        """Sets the low-pass bandpass filter frequency."""
        EXT_SYNC_SET = self.bus.read_byte_data(self.address, self.MPU_CONFIG) & 0b00111000
        self.bus.write_byte_data(self.address, self.MPU_CONFIG, EXT_SYNC_SET | filter_range)

    def read_gyro_range(self, raw=False):
        """Reads the range the gyroscope is set to."""
        raw_data = self.bus.read_byte_data(self.address, self.GYRO_CONFIG)
        if raw:
            return raw_data
        else:
            if raw_data == self.GYRO_RANGE_250DEG:
                return 250
            elif raw_data == self.GYRO_RANGE_500DEG:
                return 500
            elif raw_data == self.GYRO_RANGE_1000DEG:
                return 1000
            elif raw_data == self.GYRO_RANGE_2000DEG:
                return 2000
            else:
                return -1

    def get_gyro_data(self):
        """Gets and returns the X, Y and Z values from the gyroscope."""
        x = self.read_i2c_word(self.GYRO_XOUT0)
        y = self.read_i2c_word(self.GYRO_YOUT0)
        z = self.read_i2c_word(self.GYRO_ZOUT0)
        gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        gyro_range = self.read_gyro_range(True)
        if gyro_range == self.GYRO_RANGE_250DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_2000DEG
        x = x / gyro_scale_modifier
        y = y / gyro_scale_modifier
        z = z / gyro_scale_modifier
        return {'x': x, 'y': y, 'z': z}

    def get_all_data(self):
        """Reads and returns all the available data."""
        temp = self.get_temp()
        accel = self.get_accel_data()
        gyro = self.get_gyro_data()
        return [accel, gyro, temp]

    def calibrate(self):
        """Calibrates the sensor."""
        totalGyroX = 0
        totalGyroY = 0
        totalGyroZ = 0
        totalAccX  = 0
        totalAccY  = 0
        totalAccZ  = 0

        print("gyro calibration in process...")
        for _ in range(self.DISCARDED_MEASURES):
            gyroData = self.get_gyro_data()
            AccData = self.get_accel_data()
            time.sleep(0.001)
        for i in range(self.CALIBRATION_MEASURES):
            gyroData = self.get_gyro_data()
            AccData = self.get_accel_data()
            totalGyroX += gyroData['x']
            totalGyroY += gyroData['y']
            totalGyroZ += gyroData['z']
            totalAccX  += AccData['x']
            totalAccY  += AccData['y']
            totalAccZ  += AccData['z']
            time.sleep(0.001)
        self.gyroXoffset = totalGyroX / self.CALIBRATION_MEASURES
        self.gyroYoffset = totalGyroY / self.CALIBRATION_MEASURES
        self.gyroZoffset = totalGyroZ / self.CALIBRATION_MEASURES
        self.accXoffset = totalAccX / self.CALIBRATION_MEASURES
        self.accYoffset = totalAccY / self.CALIBRATION_MEASURES
        self.accZoffset = totalAccZ / self.CALIBRATION_MEASURES
        self.accZoffset = self.accZoffset - self.GRAVITIY_MS2
        print("gyro calibration complete")
    
    def update(self):
        accelData = self.get_accel_data(False)
        gyroData = self.get_gyro_data()

        gyroXrate = gyroData['x'] - self.gyroXoffset
        gyroYrate = gyroData['y'] - self.gyroYoffset
        gyroZrate = gyroData['z'] - self.gyroZoffset
        accXrate = accelData['x'] - self.accXoffset
        accYrate = accelData['y'] - self.accYoffset
        accZrate = accelData['z'] - self.accZoffset

        accelXangle = math.atan2(accYrate, accZrate) * self.RAD_TO_DEG
        accelYangle = math.atan2(accXrate, math.sqrt(accXrate**2 + accZrate**2)) * -self.RAD_TO_DEG

        self.intervalEnd = time.time()
        interval = self.intervalEnd - self.intervalStart
        self.intervalStart = self.intervalEnd

        self.angGyroX += gyroXrate * interval
        self.angGyroY += gyroYrate * interval
        self.angGyroZ += gyroZrate * interval
        #apply complementary filter
        self.angX = self.filterAccelCoeff * accelXangle + self.filterGyroCoeff * self.angGyroX
        self.angY = self.filterAccelCoeff * accelYangle + self.filterGyroCoeff * self.angGyroY
        self.angZ = self.angGyroZ

        #angles in radian
        phi = self.angX * self.DEG_TO_RAD
        the = self.angY * self.DEG_TO_RAD
        psi = self.angZ * self.DEG_TO_RAD
        
        accXcomp = accXrate + self.GRAVITIY_MS2 * np.sin(the) * np.cos(phi)
        accYcomp = accYrate - self.GRAVITIY_MS2 * np.sin(phi)
        accZcomp = accZrate - self.GRAVITIY_MS2 * np.cos(phi) * np.cos(the)

        self.ROT = np.array([[np.cos(psi)*np.cos(the),
                              np.sin(phi)*np.sin(the)*np.cos(psi) + np.cos(phi)*np.sin(psi),
                              -np.cos(phi)*np.sin(the)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
                             [-np.cos(the)*np.sin(psi),
                              -np.sin(phi)*np.sin(the)*np.sin(psi) + np.cos(phi)*np.cos(psi),
                              np.cos(phi)*np.sin(the)*np.sin(psi) + np.sin(phi)*np.cos(psi)],
                             [np.sin(the),
                              -np.sin(phi)*np.cos(the),
                              np.cos(phi)*np.cos(the)]])

        accel = np.transpose(self.ROT) @ np.array([[accXcomp],[accYcomp],[accZcomp]])
        if abs(accel[0]) > self.accThreshold:
            self.speed[0] += accel[0]*interval
        if abs(accel[1]) > self.accThreshold:
            self.speed[1] += accel[1]*interval
        if abs(accel[2]) > self.accThreshold:
            self.speed[2] += accel[2]*interval
        velocity = self.conv @ self.speed.reshape((3,1))                             #speed matrix converted to vehicle coordin.
        velocity = velocity.reshape(3)
        rotation = self.conv @ self.ROT @ np.transpose(self.conv)

        return {'rx': self.angX, 'ry': self.angY, 'rz': self.angZ,
                'ax': accXcomp, 'ay': accYcomp, 'az': accZcomp,
                'v': velocity, 'rot': rotation}


if __name__ == "__main__":
    mpu = mpu6050(0x68)
    mpu.calibrate()