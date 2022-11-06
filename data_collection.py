import cv2 #importing_webcam
from cvzone.HandTrackingModule import HandDetector
import numpy as np #to_make_matrix
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


offset = 20
imgSize = 300

folder = "Data/Z"   #to_save_image_to_folder
counter = 0 #to_count_images

while True:
    success, img = cap.read() #retrieve_video
    hands, img = detector.findHands(img) #detect_hand
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] #bounding_box

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #to_make_image_in background_as_matrix
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #to_crop_image


        imgCropShape = imgCrop.shape #adding_image_to_white_background

        aspectRatio = h / w

        if aspectRatio > 1: #to_get_maximum_size_image_when_h_>_w
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2) #to_centre_image_in_white_back
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w #to_get_maximum_size_image_when_w_>_h
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)







