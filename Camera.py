import cv2
import numpy as np


class Camera:

    def __init__(self):

        self.h_min = 116
        self.h_max = 133
        self.s_min = 79
        self.s_max = 255
        self.v_min = 54
        self.v_max = 214

        self.direction = 0
        self.isFiring = False

        self.cap = None
        self.image_hsv = None

    def update_segmentation(self):
        if self.h_min < self.h_max:
            _, mask_h_min = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_min,
                                         maxval=1, type=cv2.THRESH_BINARY)
            _, mask_h_max = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_max,
                                         maxval=1, type=cv2.THRESH_BINARY_INV)
            mask_h = mask_h_min * mask_h_max
        else:
            _, mask_h_min = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_min,
                                         maxval=1, type=cv2.THRESH_BINARY_INV)
            _, mask_h_max = cv2.threshold(src=self.image_hsv[:, :, 0], thresh=self.h_max,
                                         maxval=1, type=cv2.THRESH_BINARY)
            mask_h = cv2.bitwise_or(mask_h_min, mask_h_max)

        _, mask_s_min = cv2.threshold(src=self.image_hsv[:, :, 1], thresh=self.s_min,
                                     maxval=1, type=cv2.THRESH_BINARY)
        _, mask_smax = cv2.threshold(src=self.image_hsv[:, :, 1], thresh=self.s_max,
                                     maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_s = mask_s_min * mask_smax

        _, mask_v_min = cv2.threshold(src=self.image_hsv[:, :, 2], thresh=self.v_min,
                                     maxval=1, type=cv2.THRESH_BINARY)
        _, mask_v_max = cv2.threshold(src=self.image_hsv[:, :, 2], thresh=self.v_max,
                                     maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_v = mask_v_min * mask_v_max

        mask = mask_h * mask_s * mask_v
        cv2.imshow("Mask", mask * 255)

        contours, hierarchy = cv2.findContours(image=mask,
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)
        mask_filtered = np.zeros(mask.shape, dtype=np.uint8)
        for i in range(len(contours)):
            contour = contours[i]
            contour_area = cv2.contourArea(contour)
            if contour_area > 100:
                cv2.drawContours(image=mask_filtered, contours=contours,
                                 contourIdx=i, color=(1, 1, 1), thickness=-1)
                m = cv2.moments(contour)
                cx = int(np.round(m['m10'] / m['m00']))  # Center x
                cy = int(np.round(m['m01'] / m['m00']))  # Center y
                perimeter = cv2.arcLength(curve=contour, closed=True)
                if cx > (2 / 3) * mask.shape[1]:
                    cv2.rectangle(img=mask_filtered,
                                  pt1=(mask.shape[1] - 10, 0),
                                  pt2=(mask.shape[1] - 10, mask.shape[0]),
                                  color=(1, 1, 1), thickness=6)
                    cv2.imshow("Mask Filtered", mask_filtered * 255)

                elif cx < (1 / 3) * mask.shape[1]:
                    cv2.rectangle(img=mask_filtered,
                                  pt1=(10, 0),
                                  pt2=(10, mask.shape[0]),
                                  color=(1, 1, 1), thickness=6)
                    cv2.imshow("Mask Filtered", mask_filtered * 255)

                else:
                    cv2.imshow("Mask Filtered", mask_filtered * 255)

    def on_change_h_min(self, val):
        self.h_min = val

    def on_change_h_max(self, val):
        self.h_max = val

    def on_change_s_min(self, val):
        self.s_min = val

    def on_change_s_max(self, val):
        self.s_max = val

    def on_change_v_min(self, val):
        self.v_min = val

    def on_change_v_max(self, val):
        self.v_max = val

    def open_camera(self):
        self.cap = cv2.VideoCapture()
        if not self.cap.isOpened():
            self.cap.open(0)
        ret, image = self.cap.read()
        self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cv2.namedWindow("Image")
        cv2.createTrackbar("H_min", "Image", self.h_min, 180, self.on_change_h_min)
        cv2.createTrackbar("H_max", "Image", self.h_max, 180, self.on_change_h_max)
        cv2.createTrackbar("S_min", "Image", self.s_min, 255, self.on_change_s_min)
        cv2.createTrackbar("S_max", "Image", self.s_max, 255, self.on_change_s_max)
        cv2.createTrackbar("V_min", "Image", self.v_min, 255, self.on_change_v_min)
        cv2.createTrackbar("V_max", "Image", self.v_max, 255, self.on_change_v_max)

    def update_camera(self):
        if not self.cap.isOpened():
            self.cap.open(0)
        _, image = self.cap.read()
        image = image[:, ::-1, :]
        cv2.imshow("Image", image)
        self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.update_segmentation()

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
