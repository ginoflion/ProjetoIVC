from ultralytics import YOLO
import os.path
import cv2
import time
import numpy as np

class YOLOIntegration:

    def __init__(self):
        self.last_frame_timestamp = 0
        self.model = YOLO("yolov8n.pt")
        cv2.namedWindow("Image")


    def process_frame(self, image):
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        results = self.model(image, verbose=False)
        image_objects = image.copy()

        objects = results[0]
        for obj in objects:
            box = obj.boxes.data[0]
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            confidence = box[4]
            class_id = int(box[5])
            if class_id == 67 and confidence > 0.20:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                # Draw rectangles based on phone's position
                if cx > (2 / 3) * image_objects.shape[1] and (1 / 3) * image_objects.shape[0] < cy < (2 / 3) * \
                        image_objects.shape[0]:
                    cv2.rectangle(img=image_objects, pt1=(image_objects.shape[1] - 10, 0),
                                  pt2=(image_objects.shape[1], image_objects.shape[0]),
                                  color=(255, 255, 255), thickness=6)
                elif cx < (1 / 3) * image_objects.shape[1] and (1 / 3) * image_objects.shape[0] < cy < (2 / 3) * \
                        image_objects.shape[0]:
                    cv2.rectangle(img=image_objects, pt1=(0, 0), pt2=(10, image_objects.shape[0]),
                                  color=(255, 255, 255), thickness=6)
                elif cy > (2 / 3) * image_objects.shape[0] and (1 / 3) * image_objects.shape[1] < cx < (2 / 3) * \
                        image_objects.shape[1]:
                    cv2.rectangle(img=image_objects, pt1=(0, image_objects.shape[0] - 10),
                                  pt2=(image_objects.shape[1], image_objects.shape[0]),
                                  color=(255, 255, 255), thickness=6)
                elif cy < (1 / 3) * image_objects.shape[0] and (1 / 3) * image_objects.shape[1] < cx < (2 / 3) * \
                        image_objects.shape[1]:
                    cv2.rectangle(img=image_objects, pt1=(0, 0), pt2=(image_objects.shape[1], 10),
                                  color=(255, 255, 255), thickness=6)
                self.object_center_x = cx
                self.object_center_y = cy
                # Draw a red rectangle around the phone
                cv2.rectangle(img=image_objects, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])),
                              color=(255, 0, 0), thickness=2)

                text = "{}:{:.2f}".format(objects.names[class_id], confidence)
                cv2.putText(img=image_objects,
                            text=text,
                            org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 0, 0),
                            thickness=1)

        text_to_show = str(int(np.round(1 / (time.time() - self.last_frame_timestamp)))) + " fps"
        cv2.putText(img=image_objects,
                    text=text_to_show,
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1)
        self.last_frame_timestamp = time.time()

        return cv2.cvtColor(src=image_objects, code=cv2.COLOR_RGB2BGR)



    def get_object_position(self):
        return self.object_center_x, self.object_center_y



    def open_camera(self):
        self.cap = cv2.VideoCapture()
        if not self.cap.isOpened():
            self.cap.open(0)
        ret, image = self.cap.read()


    def update_camera(self):
        _, image = self.cap.read()
        image = cv2.flip(image, 1)
        processed_frame = self.process_frame(image)
        cv2.imshow(winname="Image", mat=processed_frame)


    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

