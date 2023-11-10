from ultralytics import YOLO
import os.path
import cv2
import time
import numpy as np

class YOLOIntegration:
    def __init__(self):
        self.last_frame_timestamp = 0
        self.cap = cv2.VideoCapture()
        self.model = YOLO("yolov8n.pt")
        print("Known classes ({})".format(len(self.model.names)))
        for i in range(len(self.model.names)):
            print("{} : {}".format(i, self.model.names[i]))

        cv2.namedWindow("Image")

    def start_camera(self, camera_index=0):
        if not self.cap.isOpened():
            self.cap.open(camera_index)
    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

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
            if class_id == 67 and confidence > 0.25:
                self.draw_object_indicator(image_objects, box)
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

    def draw_object_indicator(self, image, box):
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        if cx > (2 / 3) * image.shape[1] and (1 / 3) * image.shape[0] < cy < (2 / 3) * image.shape[0]:
            cv2.rectangle(img=image, pt1=(image.shape[1] - 10, 0),
                          pt2=(image.shape[1] - 10, image.shape[0]),
                          color=(255, 255, 255), thickness=6)
        elif cx < (1 / 3) * image.shape[1] and (1 / 3) * image.shape[0] < cy < (2 / 3) * image.shape[0]:
            cv2.rectangle(img=image, pt1=(10, 0), pt2=(10, image.shape[0]),
                          color=(255, 255, 255), thickness=6)
        elif cy > (2 / 3) * image.shape[0] and (1 / 3) * image.shape[1] < cx < (2 / 3) * image.shape[1]:
            cv2.rectangle(img=image, pt1=(0, image.shape[0] - 10),
                          pt2=(image.shape[1], image.shape[0] - 10),
                          color=(255, 255, 255), thickness=6)
        elif cy < (1 / 3) * image.shape[0] and (1 / 3) * image.shape[1] < cx < (2 / 3) * image.shape[1]:
            cv2.rectangle(img=image, pt1=(0, 10), pt2=(image.shape[1], 10),
                          color=(255, 255, 255), thickness=6)
        cv2.rectangle(img=image, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(255, 0, 0),
                      thickness=2)

        self.object_center_x = cx
        self.object_center_y = cy

    def get_object_position(self):
        return self.object_center_x, self.object_center_y

    def run(self):
        while True:
            if not self.cap.isOpened():
                self.cap.open(0)

            _, image = self.cap.read()
            image = cv2.flip(image, 1)

            processed_frame = self.process_frame(image)

            cv2.imshow(winname="Image", mat=processed_frame)

            c = cv2.waitKey(delay=1)
            if c == 27:
                break


if __name__ == "__main__":
    integration = YOLOIntegration()
    integration.start_camera()
    integration.run()
    integration.close_camera()
    cv2.destroyAllWindows()