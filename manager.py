import cv2, time
import numpy as np
import argparse, random
from gaze_tracking.gaze_tracking import GazeTracking
from Headpose_Detection import headpose
from pynput import keyboard

from enum import Enum
from random import uniform
from gaze_tracking.pupil import Pupil
from socketClient import socketConnect



class CalibrationStep(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class CalibrationValue:
    def __init__(self):
        self.head_move = {
            "left": None,
            "right": None,
            "up": None,
            "down": None
        }

        self.pulpil_move = {
            "left": None,
            "right": None,
            "up": None,
            "down": None
        }

        self.delta = {
            "horizontal": None,
            "vertical": None
        }   

    def calc_delta(self):
        self.delta['horizontal'] = (abs(self.head_move['right']['ry'] - self.head_move['left']['ry'])) / (abs(self.pulpil_move['right']['horizontal'] - self.pulpil_move['left']['horizontal']))
        self.delta['vertical'] = abs(self.head_move['up']['rx'] - self.head_move['down']['rx']) / abs(self.pulpil_move['up']['vertical'] - self.pulpil_move['down']['vertical'])
        
        print(self.delta)

class Manager:
    def __init__(self):
        self.calibration_value = CalibrationValue()
        self.gaze = GazeTracking()
        self.headpose = None
        self.camera = None
        
        self.pupil_cheat = False
        # 

    def start_camera_capture(self, args):
        cap = cv2.VideoCapture(0)
        cap.set(3, args['wh'][0])
        cap.set(4, args['wh'][1])
        self.camera = cap

    def start_camera_open(self):
        ret, frame = self.camera.read()
    

    def initialize_headpose(self, args):
        self.headpose = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])

    def make_calibration_screen(self, step, movement="Head"):
        background = np.zeros((480, 720, 3), np.uint8)
        background.fill(255)
        background = cv2.putText(background, f"Look at the point and press space", (76, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)
        background = cv2.putText(background, f"({movement})", (300 , 280), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

        if step == CalibrationStep.UP:
            background = cv2.circle(background, (360,12), 1, 255, 24)

        elif step == CalibrationStep.RIGHT:
            background = cv2.circle(background, (708,240), 1, 255, 24)

        elif step == CalibrationStep.DOWN:
            background = cv2.circle(background, (360,468), 1, 255, 24)

        else:
            background = cv2.circle(background, (12,240), 1, 255, 24)
            
        return background
    
    def make_configuration(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
        parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
        parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
        parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
        parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                            default='./model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
        args = vars(parser.parse_args())

        return args

    # 사용자 동공 및 머리 위치 값을 가져옴
    def get_current_value(self):
        ret, camera = self.camera.read()
        
        camera = cv2.flip(camera, 1)
        camera, angles = self.headpose.process_image(camera)

        self.gaze.refresh(camera)
        # 좌우 방향에 대한 값. -0.5(right) ~ 0.5(left) 사이의 값
        horizontal_pulpil_ratio = self.gaze.horizontal_ratio() - 0.5
        # 상하 방향에 대한 값. -0.5(top) ~ 0.5(bottom) 사이의 값
        vertical_pulpil_ratio = self.gaze.vertical_ratio() - 0.5
        value = {
            "rx": angles[0],
            "ry": angles[1],
            "rz": angles[2],
            "vertical": vertical_pulpil_ratio,
            "horizontal": horizontal_pulpil_ratio
        }
        
        print(f"""
Headpose rx : {angles[0]}
Headpose ry : {angles[1]}
Headpose rz : {angles[2]}
Gaze Horizontal : {horizontal_pulpil_ratio}
Gaze Vertical : {vertical_pulpil_ratio}
        """)

        return value
        
    def process_calibration_value(self, step, movement):
        target_value = self.calibration_value.head_move if movement == "Head" else self.calibration_value.pulpil_move

        value = self.get_current_value()
        print("value :", value)

        if step == CalibrationStep.UP:
            target_value['up'] = value
        elif step == CalibrationStep.LEFT:
            target_value['left'] = value
        elif step == CalibrationStep.DOWN:
            target_value['down'] = value
        else:
            target_value['right'] = value

        print("target_value :", target_value)

    def calibration(self):
        movement = "Head"
        
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.UP))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.UP, movement)
        
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.RIGHT))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.RIGHT, movement)

        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.DOWN))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.DOWN, movement)

        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.LEFT))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.LEFT, movement)

        movement = "Pupil"

        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.UP, movement))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.UP, movement)
        
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.RIGHT, movement))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.RIGHT, movement)

        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.DOWN, movement))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.DOWN, movement)

        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.LEFT, movement))
        cv2.waitKey()
        self.process_calibration_value(CalibrationStep.LEFT, movement)

        self.calibration_value.calc_delta()

    def is_in_monitor(self):
        try:
            weight = 0.3
            value = self.get_current_value()
            delta = self.calibration_value.delta

            horizontal_vector = value['ry'] + ((value['horizontal'] * delta['horizontal']) * weight)
            vertical_vector = value['rx']

            print(f"""
horizontal_vector : {horizontal_vector}
vertical_vector : {vertical_vector}
            """)

            limit_left = self.calibration_value.head_move['left']['ry']
            limit_right = self.calibration_value.head_move['right']['ry']
            limit_up = self.calibration_value.head_move['up']['rx']
            limit_down = self.calibration_value.head_move['down']['rx']

            print(f"""
limit_left : {limit_left}
limit_right : {limit_right}
limit_up : {limit_up}
limit_down : {limit_down}
            """)

            if limit_left <= horizontal_vector <= limit_right and limit_down <= vertical_vector <= limit_up:
                print("모니터 내부")
            else:
                print("모니터 외부")
                self.pupil_cheat = True

        except Exception as e:
            print(e)
            print("값 검출 에러")
        

    def get_pupil_value(self):
        left_x, left_y = self.gaze.pupil_left_coords()
        right_x, right_y = self.gaze.pupil_right_coords()
        return left_x, left_y, right_x, right_y


    def start(self):

        while True:
            cv2.waitKey()
            self.is_in_monitor()

            left_x, left_y, right_x, right_y = self.get_pupil_value()
            print("left_x :",left_x)
            obj = {
                'left_x': int(left_x), 
                'left_y': int(left_y), 
                'right_x' : int(right_x),
                'right_y' : int(right_y),
                'cheat' : self.pupil_cheat
            }

            socketConnect(obj)

    def video(self):
        ret, frame = self.camera.read()
        cv2.imshow('frame', frame)


    def run(self):

        # configuration = self.make_configuration()

        # self.start_camera_capture(configuration)
        # self.initialize_headpose(configuration)
        # self.calibration()    

        # self.start()


        while True:
            
            l_x = uniform(10, 24)
            l_y = uniform(5,50)
            r_x = uniform(20, 44)
            r_y = uniform(6,50)
            ch = random.choice([True, False])
            obj = {
                    'left_x': l_x, 
                    'left_y': l_y, 
                    'right_x' : r_x,
                    'right_y' : r_y,
                    'cheat' : ch
                }

            socketConnect(obj)
                
    
        #calibration한 후 값을 받아오는 작업을 해야겠움

