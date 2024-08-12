# !usr/bin/env python3

import serial
import struct
import time
import numpy as np
import serial.tools.list_ports as ports

class GripperData:
    def __init__(self):
        # ------------------------ CONSTANTS ------------------------ #
        self.PMAX = 13 # which maps to about 1.4 bars
        self.arduinoPB = None

    def set_communication(self, aportPB, baudrate):
        try:
            self.arduinoPB = serial.Serial( 
                                port=aportPB,
                                baudrate=baudrate,
                                timeout=0.1 
                                )
            self.arduinoPB.isOpen() # try to open port
            print ("Valve is opened!")
        except IOError: # if port is already opened, close it and open it again and print message
            self.arduinoPB.close()
            self.arduinoPB.open()
            print ("Valve was already open, was closed and opened again!")

    ## Send values to the Pressure Box
    def send_values(self, U1, U2, U3, L1, L2, L3, G, PMAX):
        if U1 > PMAX or U2 > PMAX or U3 > PMAX or L1 > PMAX or L2 > PMAX or L3 > PMAX or G > 1:
            print("Pressure out of range")
            return

        packet = np.array([106,U1,U2,U3,L1,L2,L3,G],dtype = np.uint8)
        
        if self.arduinoPB.isOpen():
            for value in packet : 
                s = struct.pack('!{0}B'.format(len(packet)), *packet)
                self.arduinoPB.write(s)

    def actuate_finger(self, cycles, pressure_value, aportPB, PMAX):
        try:
            start_time = time.time()
            for cycle in range(cycles):
                self.set_communication(aportPB=aportPB) # open the port
                self.send_values(pressure_value,0,0,0,0,0,0,PMAX)  # Open the valve V1 and grasp the object
                self.send_values(0,0,0,0,0,0,0,PMAX) # Release the object
                self.arduinoPB.close() # close the valve
    
            end_time = time.time()
            total_time = (end_time - start_time)
            print(f"Total time taken for the experiment is {total_time}")

        except Exception as e:
            print(f"Error encountered while sending value to control Box, closing the port!!")
            self.send_values(0,0,0,0,0,0,0,PMAX) # Release the object
            self.arduinoPB.close()
