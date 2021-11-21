from imutils.video import FPS
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import cv2 # opencv 모듈
import os

from yolov3 import yolo # 운영체제 기능 모듈


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
        self.totalFrame = 0
        self.vs = ""
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

    def stream(self, ln):
        (W, H) = (None, None)
        while True:
            # 프레임 읽기
            ret, frame = self.vs.read()

            # 읽은 프레임이 없는 경우 종료
            # if self.args["input"] is not None and frame is None:
            #     break
            
            # 프레임 크기 지정
            frame = imutils.resize(frame, width=800, height=400)

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
                        
                        # boun화ding box 왼쪽 위 좌표
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
    
    def run(self, ln):
        fps = FPS().start()

        self.vs = cv2.VideoCapture(0)
        print("[webcam 시작]")

        self.stream(ln)
        
        # fps 정보 업데이트
        fps.update()

        fps.stop()
        print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
        print("[FPS : {:.2f}]".format(fps.fps()))

        # 종료
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    y3 = Yolo()
    y3.configure()
    ln = y3.get_layer()
    y3.run(ln)