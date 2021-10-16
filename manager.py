import cv2
import numpy as np
import argparse
from gaze_tracking.gaze_tracking import GazeTracking
from Headpose_Detection import headpose

from enum import Enum

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

    def start_camera_capture(self, args):
        cap = cv2.VideoCapture(0)
        cap.set(3, args['wh'][0])
        cap.set(4, args['wh'][1])
        self.camera = cap

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

    def process_calibration_value(self, step, movement):
        target_value = self.calibration_value.head_move if movement == "Head"else self.calibration_value.pulpil_move

        ret, camera = self.camera.read()
        
        camera = cv2.flip(camera, 1)
        camera, angles = self.headpose.process_image(camera)

        self.gaze.refresh(camera)
        horizontal_pulpil_ratio = self.gaze.horizontal_ratio()
        vertical_pulpil_ratio = self.gaze.vertical_ratio()
        value = {
            "rx": angles[0],
            "ry": angles[1],
            "rz": angles[2],
            "vertical": vertical_pulpil_ratio,
            "horizontal": horizontal_pulpil_ratio
        }
        
        print(f"""
캘리브레이션 값 캡쳐
Headpose rx : {angles[0]}
Headpose ry : {angles[1]}
Headpose rz : {angles[2]}
Gaze Horizontal : {horizontal_pulpil_ratio}
Gaze Vertical : {vertical_pulpil_ratio}
        """)

        if step == CalibrationStep.UP:
            target_value['up'] = value
        elif step == CalibrationStep.LEFT:
            target_value['left'] = value
        elif step == CalibrationStep.DOWN:
            target_value['down'] = value
        else:
            target_value['right'] = value

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
         
    def run(self):
        configuration = self.make_configuration()

        self.start_camera_capture(configuration)
        self.initialize_headpose(configuration)
        self.calibration()

