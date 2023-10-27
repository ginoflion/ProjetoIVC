import cv2
import numpy as np

hmin = 0
hmax = 180
smin = 0
smax = 255
vmin = 0
vmax = 255


def update_segmentation(image_hsv):
    global hmin, hmax, smin, smax, vmin, vmax

    if hmin < hmax:
        ret, mask_hmin = cv2.threshold(src=image_hsv[:,:,0],
                                   thresh=hmin, maxval=1, type=cv2.THRESH_BINARY)
        ret, mask_hmax = cv2.threshold(src=image_hsv[:, :, 0],
                                   thresh=hmax, maxval=1, type=cv2.THRESH_BINARY_INV)

        mask_h = mask_hmin * mask_hmax
    else:
        ret, mask_hmin = cv2.threshold(src=image_hsv[:,:,0],
                                   thresh=hmin, maxval=1, type=cv2.THRESH_BINARY_INV)
        ret, mask_hmax = cv2.threshold(src=image_hsv[:, :, 0],
                                   thresh=hmax, maxval=1, type=cv2.THRESH_BINARY)

        mask_h = cv2.bitwise_or(mask_hmin, mask_hmax)

    ret, mask_smin = cv2.threshold(src=image_hsv[:, :, 1],
                                   thresh=smin, maxval=1, type=cv2.THRESH_BINARY)
    ret, mask_smax = cv2.threshold(src=image_hsv[:, :, 1],
                                   thresh=smax, maxval=1, type=cv2.THRESH_BINARY_INV)
    mask_s = mask_smin * mask_smax

    ret, mask_vmin = cv2.threshold(src=image_hsv[:, :, 2],
                                   thresh=vmin, maxval=1, type=cv2.THRESH_BINARY)
    ret, mask_vmax = cv2.threshold(src=image_hsv[:, :, 1],
                                   thresh=vmax, maxval=1, type=cv2.THRESH_BINARY_INV)
    mask_v = mask_vmin * mask_vmax

    mask = mask_s * mask_h * mask_v
    cv2.imshow("Mask", mask * 255)

    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    mask_filtered = np.zeros(mask.shape, np.uint8)
    for i in range(len(contours)):
        contour = contours[i]
        contour_area = cv2.contourArea(contour)
        if contour_area > 100:
            cv2.drawContours(image=mask_filtered, contours=contours,
                             contourIdx=i, color=1, thickness=-1)
            M = cv2.moments(contour)
            Cx = int(np.round(M['m10'] / M['m00']))
            Cy = int(np.round(M['m01'] / M['m00']))
            perimeter = cv2.arcLength(curve=contour, closed=True)

            if Cx > mask.shape[1] / 2:
                cv2.rectangle(img=mask_filtered, pt1=(mask.shape[1] - 10, 0),
                              pt2=(mask.shape[1], mask.shape[0]), color=1, thickness=5)
            else:
                cv2.rectangle(img=mask_filtered, pt1=(0, 0),
                              pt2=(10, mask.shape[0]), color=1, thickness=5)
    cv2.imshow("Mask Filtered", mask_filtered * 255)

def on_change_hmin(val):
    global hmin
    hmin = val

def on_change_hmax(val):
    global hmax
    hmax = val

def on_change_smin(val):
    global smin
    smin = val

def on_change_smax(val):
    global smax
    smax = val

def on_change_vmin(val):
    global vmin
    vmin = val

def on_change_vmax(val):
    global vmax
    vmax = val

cv2.namedWindow("Image")
cv2.createTrackbar("Hmin", "Image", 0, 180, on_change_hmin)
cv2.createTrackbar("Hmax", "Image", 180, 180, on_change_hmax)
cv2.createTrackbar("Smin", "Image", 0, 255, on_change_smin)
cv2.createTrackbar("Smax", "Image", 255, 255, on_change_smax)
cv2.createTrackbar("Vmin", "Image", 0, 255, on_change_vmin)
cv2.createTrackbar("Vmax", "Image", 255, 255, on_change_vmax)
def image_capture():


    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        update_segmentation(image_hsv)
        cv2.imshow("Image", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
cv2.destroyAllWindows()