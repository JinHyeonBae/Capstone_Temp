import cv2, time, os
import numpy as np
import argparse, random, imutils
from gaze_tracking.gaze_tracking import GazeTracking
from Headpose_Detection import headpose
from pynput import keyboard

from enum import Enum
from random import uniform
from gaze_tracking.pupil import Pupil
#from socketClient import socketConnect
from y3 import Yolo

from threading import Thread

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

    def get_layer(self):
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return ln


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

    
    def get_layer(self):
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return ln


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
        (W, H) = (None, None)
        while True:
            # 프레임 읽기
            ret, frame = self.camera.read()

            # 읽은 프레임이 없는 경우 종료
            # if self.args["input"] is not None and frame is None:
            #     break
            
            # 프레임 크기 지정
            frame = imutils.resize(frame, width=720, height=480)

            # 프레임 크기
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            
            # 전체 프레임 수 1 증가
            self.totalFrame += 1

            # 5개의 프레임 마다 객체 인식
            if self.totalFrame % 5 == 1:
                # blob 이미지 생성
                # 파라미터
                # 1) image : 사용할 이미지
                # 2) scalefactor : 이미지 크기 비율 지정
                # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            
                # 객체 인식
                self.net.setInput(blob)
                layerOutputs = self.net.forward(ln)

                # bounding box, 확률 및 클래스 ID 목록 초기화
                boxes = []
                confidences = []
                classIDs = []
            
            # counting 수 초기
            count = 0
            
            # layerOutputs 반복
            for output in layerOutputs:
                # 각 클래스 레이블마다 인식된 객체 수 만큼 반복
                for detection in output:
                    # 인식된 객체의 클래스 ID 및 확률 추출
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    
                    # 사람이 아닌 경우 제외
                    if classID != 0: # # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                        continue
                    
                    # 객체 확률이 최소 확률보다 큰 경우
                    if confidence > self.args["confidence"]:
                        # bounding box 위치 계산
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int") # (중심 좌표 X, 중심 좌표 Y, 너비(가로), 높이(세로))
                        
                        # bounding box 왼쪽 위 좌표
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        # bounding box, 확률 및 클래스 ID 목록 추가
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        
            # bounding box가 겹치는 것을 방지(임계값 적용)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"], self.args["threshold"])
            
            # 인식된 객체가 있는 경우
            if len(idxs) > 0:
                # 모든 인식된 객체 수 만큼 반복
                for i in idxs.flatten():
                    # counting 수 증가
                    count += 1

                    # bounding box 좌표 추출
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    
                    # bounding box 출력
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 클래스 ID 및 확률
                    text = "{} : {:.2f}%".format(self.LABELS[classIDs[i]], confidences[i])

                    # label text 잘림 방지
                    y = y - 15 if y - 15 > 15 else y + 15
                    
                    # text 출력
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # counting 결과 출력
            counting_text = "People Counting : {}".format(count)
            cv2.putText(frame, counting_text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

            # 프레임 출력
            cv2.imshow("Real-Time Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' 키를 입력하면 종료
            if key == ord("q"):
                break


    def get_pupil_value(self):
        left_x, left_y = self.gaze.pupil_left_coords()
        right_x, right_y = self.gaze.pupil_right_coords()
        return left_x, left_y, right_x, right_y


    def start(self):
        # 카메라 start
        self.start_camera_capture(self.args)

        ln = self.yolo.get_layer()
        
        while True:
            try:  
                cv2.waitKey()
                self.is_in_monitor()
                self.stream(ln)
                
            
            except KeyboardInterrupt : 
                print("Interrupt!")
                break


    def run(self):

        configuration = self.make_configuration()

        self.start_camera_capture(configuration)
        self.initialize_headpose(configuration)
        self.calibration()    

        self.start()
        
     


#  left_x, left_y, right_x, right_y = self.get_pupil_value()
#             print("left_x :",left_x)
#             obj = {
#                 'left_x': int(left_x), 
#                 'left_y': int(left_y), 
#                 'right_x' : int(right_x),
#                 'right_y' : int(right_y),
#                 'cheat' : self.pupil_cheat
#             }

#             socketConnect(obj)