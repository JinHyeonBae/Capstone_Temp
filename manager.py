import cv2
import numpy as np

from enum import Enum

class CalibrationStep(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Manager:
    def __init__(self):
        self.calibration = False

    def make_calibration_screen(self, step):
        background = np.zeros((480, 720, 3), np.uint8)
        background.fill(255)
        background = cv2.putText(background, "Look at the point and press space", (76, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)


        if step == CalibrationStep.UP:
            background = cv2.circle(background, (360,12), 1, 255, 24)
        elif step == CalibrationStep.RIGHT:
            background = cv2.circle(background, (708,240), 1, 255, 24)

        elif step == CalibrationStep.DOWN:
            background = cv2.circle(background, (360,468), 1, 255, 24)

        else:
            background = cv2.circle(background, (12,240), 1, 255, 24)
            
        return background

    def run(self):
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.UP))
        cv2.waitKey()
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.RIGHT))
        cv2.waitKey()
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.DOWN))
        cv2.waitKey()
        cv2.imshow("gotcha",self.make_calibration_screen(CalibrationStep.LEFT))
        cv2.waitKey()
