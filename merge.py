

from __future__ import division

"시선 인식"
import cv2
from gaze_tracking import GazeTracking

"머리 위치 인식"
import argparse
import cv2
import numpy as np
import os.path as osp
import os

from HeadposeDetection import headpose
from gaze_tracking.pupil import Pupil

"from pupil import Pupil"

def main(args):
    filename = args["input_file"]

    gaze = GazeTracking()

    if filename is None:
        isVideo = False
        "카메라 구동"
        cap = cv2.VideoCapture(0)
        cap.set(3, args['wh'][0])
        cap.set(4, args['wh'][1])
    else:
        isVideo = True
        #비디오 파일 재생
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Initialize head pose detection
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])

    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        print('\rframe: %d' % count, end='')
        ret, frame = cap.read()
        "cap.read() => 재생되는 비디오의 한 프레임씩 읽음, 반환 bool"
        
        #inputVideo가 있으면
        if isVideo:
            frame, angles = hpd.process_image(frame)
            if frame is None: 
                break
            else:
                #동영상 촬영
                out.write(frame)
        else:
            "좌우 반전"
            frame = cv2.flip(frame, 1)
            
            "hpd => 머리 detection"
            frame, angles = hpd.process_image(frame)

            gaze.refresh(frame)
            frame = gaze.annotated_frame()
            text = ""

            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"

            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            # Display the resulting frame

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                headpose.t.summary()
                break

        if cv2.waitKey(1) == 27:
            break
        
        count += 1

    # When everything done, release the capture
    cap.release()
    if isVideo: out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='./model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    main(args)
