import cv2
import imutils
from YoloDetTRT import YoloTRT

import Jetson.GPIO as GPIO
from door_control import DoorControl
import io_control as io

from enum import Enum, auto
import threading

from http_server import Initialize_Server, Shutdown_Server, set_latest_frame, set_door_controller_reference, Fetch_Queued_Command
from ruleset_decider import RulesetDecider
from gate_states import State
from json_config import JsonConfig

import signal
import sys
import time 

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=2  
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def cleanup():
    print("[+] Cleaning up resources...")
    io.all_pins_off()
    GPIO.cleanup()
    Shutdown_Server(web_server)

def signal_handler(sig, frame):
    print('[+] Ctrl+C Detected... Exiting...')
    cleanup()
    sys.exit(0)

def main():
    #Global HTTP server for resource allocation and deallocation
    global web_server

    #Set up signal handler keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    #Initialize configuration settings for the SmartGate
    config = JsonConfig()

    #Grab respective configurations from config.json file
    model_config  = config.get_model_config() 
    rules_config  = config.get_rules_config() 
    server_config = config.get_server_config()

    #Initialize YOLOv5 model via TensorRT engine
    model = YoloTRT(model_config)

    #In the DECIDE state, the RulesetDecider will be responsible for setting the next state depending on the configuration
    decider = RulesetDecider(rules_config)

    #Start web server on a separate thread
    #Should also make the web server optional as well
    web_server = Initialize_Server(server_config)

    #Set up the GPIO channel
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(7, GPIO.OUT, initial=GPIO.LOW)

    #Initialize IO pins and door control
    io.set_all_pins()
    door_controller = DoorControl()

    #The HTTP Server would need the reference of the door_controller object to get status on each '/status' GET request
    set_door_controller_reference(door_controller)

    #Initialize State Machine
    current_state = State.IDLE

    #Initialize object detection class list
    object_list = []

    print("[+] Opening camera...")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Camera failed to open. Retrying in 3 seconds...")
        time.sleep(3)
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[-] Camera could not be opened. Check connection and drivers.")
        io.all_pins_off()
        GPIO.cleanup()
        sys.exit(1)

    print("[+] Camera opened successfully.")

    #Our main loop
    while True:
        #----------Check for commands from POST requests------------
        command = Fetch_Queued_Command()
        if command:
            if command == 'OPEN_DOOR':
                current_state = State.DOOR_OPEN
            elif command == 'CLOSE_DOOR':
                current_state = State.DOOR_CLOSE

        #------------IDLE State ------------------------------------
        if current_state == State.IDLE:
            print("System is idle.")

            if door_controller.is_door_fully_closed() and door_controller.is_door_closing:
                door_controller.stop_door()
                print("Door fully closed, stopping motor.")
            elif door_controller.is_door_fully_open() and door_controller.is_door_opening:
                door_controller.stop_door()
                print("Door fully open, stopping motor.")

            if io.get_val('PIR'):
                current_state = State.DETECT
            else:
                current_state = State.IDLE

        #------------DETECT State ----------------------------------
        elif current_state == State.DETECT:
            print("Detecting objects.")
            ret_val, img = cap.read()
            if not ret_val or img is None:
                print("[!] Failed to read frame, retrying...")
                time.sleep(0.1)
                current_state = State.IDLE
                continue

            img = imutils.resize(img, width=600)
            detections, t = model.Inference(img)
            set_latest_frame(img.copy())
            object_list = [obj['class'] for obj in detections]
            current_state = State.DECISION

        #------------DECISION State --------------------------------
        elif current_state == State.DECISION:
            print("Decision making door.")
            current_state = decider.decide(object_list)

        #------------DOOR OPEN State -------------------------------
        elif current_state == State.DOOR_OPEN:
            print("Opening door.")
            if not door_controller.is_door_fully_open():
                door_controller.open_door()
            else:
                print('Door stopped on opening')
                door_controller.stop_door()
            current_state = State.IDLE

        #------------DOOR CLOSE State ------------------------------
        elif current_state == State.DOOR_CLOSE:
            print("Closing door.")
            if not door_controller.is_door_fully_closed():
                door_controller.close_door()
            else:
                print('Door stopped on closing')
                door_controller.stop_door()
            current_state = State.IDLE

        #------------Default State ---------------------------------
        elif current_state == State.DELAY:
            print("Delaying operation.")

    #Sets all pins to LOW
    io.all_pins_off()

#Main logic
if __name__ == "__main__":
    main()
