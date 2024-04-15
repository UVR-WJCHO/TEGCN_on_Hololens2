import time
import cv2
import numpy as np
from handtracker.module_SARTE import HandTracker

track_hand = HandTracker()

sample_input = cv2.imread('./handtracker/00000023.jpg')
_ = track_hand.Process_single_nomp(sample_input, flag_vis=True, flag_demo=True)
cv2.waitKey(0)