from threading import Thread
from imutils.video import FPS
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import cv2 # opencv 모듈
import os
import ctypes

from queue import Queue
import schedule
from socketClient import socketClient

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
print(screensize)

#from socketClient import socketConnect

writer = None
(W, H) = (None, None)

class Yolo:

    def __init__(self) -> None:
        # YOLO 모델이 학습된 coco 클래스 레이블
        self.labelsPath = os.path.sep.join(["model", "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        # YOLO 가중치 및 모델 구성에 대한 경로
        self.weightsPath = os.path.sep.join(["model", "yolov3.weights"]) # 가중치
        self.configPath = os.path.sep.join(["model", "yolov3.cfg"]) # 모델 구성

        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.detect_obj = []
        self.detectionSocket = socketClient()

        self.totalFrame = 0
        self.person_count = 0
        self.obj = ""
        self.objQueue = Queue()
        self.args = ""
    
    #arg 정의
    def configure(self):
        # 실행을 할 때 인자값 추가
        ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
        # 입력받을 인자값 등록
        ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
        ap.add_argument("-o", "--output", type=str, help="output 비디오 경로") # 비디오 저장 경로
        ap.add_argument("-c", "--confidence", type=float, default=0.5, help="최소 확률")
        # 퍼셉트론 : 입력 값과 활성화 함수를 사용해 출력 값을 다음으로 넘기는 가장 작은 신경망 단위
        # 입력 신호가 뉴런에 보내질 때 가중치가 곱해짐
        # 그 값들을 더한 값이 한계값을 넘어설 때 1을 출력
        # 이 때 한계값을 임계값이라고 함
        ap.add_argument("-t", "--threshold", type=float, default=0.3, help="임계값")
        # 입력받은 인자값을 args에 저장
        self.args = vars(ap.parse_args())
        

    def get_layer(self):
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return ln

    def object_detection(self, frame, ln):

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
        return layerOutputs

    def calc_object_rate(self, layerOutputs, shape):
        self.configure()
        # 프레임 크기
        (H, W) = shape[:2]
        
        #Y축, X축, 채널의 수
        boxes = []
        confidences = []
        classIDs = []

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
    
        return boxes, confidences, classIDs

    def draw_object(self, idxs : list, boxes, confidences, classIDs, frame):
        #counting 수 초기
        count = 0
        self.detect_obj = [] # 초기화
        # 인식된 객체가 있는 경우
        if len(idxs) > 0:
            
            # 모든 인식된 객체 수 만큼 반복
            for i in idxs.flatten():
                # counting 수 증가
                count += 1

                # bounding box 좌표 추출
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
        
                self.person_count = count
                self.obj = self.make_shape(x,y,w,h)
                self.detect_obj.append(self.obj)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 클래스 ID 및 확률
                text = "{} : {:.2f}%".format(self.LABELS[classIDs[i]], confidences[i])

                # label text 잘림 방지
                y = y - 15 if y - 15 > 15 else y + 15
                 
                # text 출력
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else : 
            self.person_count = 0
            self.obj = self.make_shape()
            self.detect_obj.append(self.obj)

        # counting 결과 출력

        return frame
    
    camera = cv2.VideoCapture(0)
    def stream(self, ln):
        schedule.every(3).seconds.do(self.send)
        while True:
            # 프레임 읽기
            schedule.run_pending()
            ret, frame = self.vs.read()
            
            # 프레임 크기 지정
            resizerate = screensize[0] / screensize[1]
            frame = cv2.resize(frame, (0,0), fx=resizerate, fy=1)
           
            # 전체 프레임 수 1 증가
            self.totalFrame += 1
            # 프레임 별로 object_detection하므로 나누기 수를 늘리면 출력 영상의 속도는 빠름
            if self.totalFrame % 5 == 1:
                layerOutputs = self.object_detection(frame, ln)

            boxes, confidences, classIDs = self.calc_object_rate(layerOutputs, frame.shape)
                    
            # bounding box가 겹치는 것을 방지(임계값 적용)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"], self.args["threshold"])
            
            frame = self.draw_object(idxs, boxes,confidences, classIDs, frame)
            
            #socket send

            # 프레임 출력
            cv2.imshow("Real-Time Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            # 'q' 키를 입력하면 종료
            if key == ord("q"):
                break

    
    def send(self):
        print("send :", self.detect_obj)
        self.detectionSocket.send(self.detect_obj, 'localhost', 8082)


    def start_camera(self):
        self.vs = cv2.VideoCapture(0)


    def run(self):
        fps = FPS().start()

        ln = self.get_layer()
        self.start_camera()
        self.stream(ln)

        # fps 정보 업데이트
        fps.update()

        fps.stop()
        print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
        print("[FPS : {:.2f}]".format(fps.fps()))

        # 종료
        # self.vs.release()
        # cv2.destroyAllWindows()

    def make_shape(self, x=0, y=0, w=0, h=0):
        # 보낼 데이터는 x,y x+w, y+h, person count
        obj = {
            'x' : x,
            'y' : y,
            'w' : w,
            'h' : h,
            'number_of_person' : self.person_count,
            'type' : 'person'
        }
        print("x :", x)

        return obj

    def get_coordination(self):
        return self.obj

if __name__ == '__main__':
    y3 = Yolo()
  
    # sendThread = Thread(target=y3.send())
    
    ln = y3.get_layer()

    y3.run()
    