import cv2, time, os
import numpy as np
import argparse, random, imutils
from gaze_tracking.gaze_tracking import GazeTracking
from Headpose_Detection import headpose

from enum import Enum
from random import uniform
from gaze_tracking.pupil import Pupil
from socketClient import socketConnect
from y3 import Yolo

from threading import Thread
import ctypes

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
print(screensize)


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
        self.yolo = Yolo()
        self.headpose = None
        self.camera = None
        
        self.pupil_cheat = False
        self.current_value = ""
        self.totalFrame =0 
        # 
        self.labelsPath = os.path.sep.join(["model", "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")


        self.weightsPath = os.path.sep.join(["model", "yolov3.weights"]) # 가중치
        self.configPath = os.path.sep.join(["model", "yolov3.cfg"]) # 모델 구성

        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        

    def start_camera_capture(self, args):
        cap = cv2.VideoCapture(0)
        cap.set(3, args['wh'][0])
        cap.set(4, args['wh'][1])
        self.camera = cap

    def stop_camera_capture(self):
        self.camera.release()
        cv2.destroyAllWindows()

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
        parser.add_argument("-c", "--confidence", type=float, default=0.5, help="최소 확률")
        # 퍼셉트론 : 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 단위
        # 입력 신호가 뉴런에 보내질 때 가중치가 곱해짐
        # 그 값들을 더한 값이 한계값을 넘어설 때 1을 출력
        # 이 때 한계값을 임계값이라고 함
        parser.add_argument("-t", "--threshold", type=float, default=0.3, help="임계값")
        # 입력받은 인자값을 args에 저장
        
        self.args = vars(parser.parse_args())

        return self.args


    # 사용자 동공 및 머리 위치 값을 가져옴
    def get_current_value(self, complete = False):

        ret, frame = self.camera.read()

        camera = cv2.flip(frame, 1)
        camera, angles = self.headpose.process_image(frame)

        # camera..?
        self.gaze.refresh(camera)

        # 좌우 방향에 대한 값. -0.5(right) ~ 0.5(left) 사이의 값
        horizontal_pulpil_ratio = self.gaze.horizontal_ratio() - 0.5
        # 상하 방향에 대한 값. -0.5(top) ~ 0.5(bottom) 사이의 값
        vertical_pulpil_ratio = self.gaze.vertical_ratio() - 0.5

        self.current_value = {
            "rx": angles[0],
            "ry": angles[1],
            "rz": angles[2],
            "vertical": vertical_pulpil_ratio,
            "horizontal": horizontal_pulpil_ratio
        }

        
        return self.current_value
        
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
            
        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.UP))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.UP, movement)
                break
            except Exception as e:
                print(e)
            
        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.RIGHT))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.RIGHT, movement)
                break
            except:
                pass
        
        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.DOWN))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.DOWN, movement)
                break
            except:
                pass

        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.LEFT))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.LEFT, movement)
                break
            except:
                pass

        movement = "Pupil"

        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.UP, movement))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.UP, movement)
                break
            except:
                pass
        
        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.RIGHT, movement))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.RIGHT, movement)
                break
            except:
                pass

        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.DOWN, movement))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.DOWN, movement)
                break
            except:
                pass

        while 1:
            try:
                cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.LEFT, movement))
                cv2.waitKey()
                self.process_calibration_value(CalibrationStep.LEFT, movement)
                break
            except:
                pass
        
        self.stop_camera_capture()
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
    
    def stream(self, ln):

        # 프레임 읽기
        ret, frame = self.camera.read()

        # 읽은 프레임이 없는 경우 종료
        # if self.args["input"] is not None and frame is None:
        #     break
        
        # 프레임 크기 지정
        
        resizerate = screensize[0] / screensize[1]
        frame = cv2.resize(frame, (0,0), fx=resizerate, fy=1)
        
        # 전체 프레임 수 1 증가
        self.totalFrame += 1
        # 프레임 별로 object_detection하므로 나누기 수를 늘리면 출력 영상의 속도는 빠름
        if self.totalFrame % 5 == 1:
            layerOutputs = self.yolo.object_detection(frame, ln)

            boxes, confidences, classIDs = self.yolo.calc_object_rate(layerOutputs, frame.shape)
                    
        # bounding box가 겹치는 것을 방지(임계값 적용)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"], self.args["threshold"])
            
            frame = self.yolo.draw_object(idxs, boxes,confidences,classIDs, frame)

        # 프레임 출력
        cv2.imshow("Real-Time Object Detection", frame)


    def get_pupil_value(self):
        left_x, left_y = self.gaze.pupil_left_coords()
        print("left _x :", left_x)
        right_x, right_y = self.gaze.pupil_right_coords()
        return left_x, left_y, right_x, right_y


    def start(self):
        # 카메라 start
        #self.start_camera_capture(self.args)

        
        # left_x, left_y, right_x, right_y = self.get_pupil_value()
        # print("left_x :",left_x)
        # obj = {
        #     'left_x': int(left_x), 
        #     'left_y': int(left_y), 
        #     'right_x' : int(right_x),
        #     'right_y' : int(right_y),
        #     'cheat' : self.pupil_cheat
        # }
    
     
        
        # while True:
            
        #     l_x = uniform(10, 24)
        #     l_y = uniform(5,50)
        #     r_x = uniform(20, 44)
        #     r_y = uniform(6,50)
        #     ch = random.choice([True, False])
        #     obj = {
        #             'left_x': l_x, 
        #             'left_y': l_y, 
        #             'right_x' : r_x,
        #             'right_y' : r_y,
        #             'cheat' : ch
        #         }

        #     socketConnect(obj, 'pupil')

        
        self.start_camera_capture(self.args)
        ln = self.yolo.get_layer()

        while True:
            try: 
                self.is_in_monitor()
                self.stream(ln)
                
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

            except KeyboardInterrupt : 
                print("Interrupt!")
                break


    def run(self):

        configuration = self.make_configuration()

        self.start_camera_capture(configuration)
        self.initialize_headpose(configuration)
        self.calibration()    

        self.start()