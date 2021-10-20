import cv2
import os
import threading
from uuid import uuid1
import argparse

class DataCollection:
    blocked_dir = './dataset/blocked'
    free_dir = './dataset/free'
    blocked_count = len(os.listdir(blocked_dir))
    free_count = len(os.listdir(free_dir))

    def __init__(self, width=224, height=224,
                 capture_width=640, capture_height=480):
        self.width = width
        self.height = height
        self.cap_width = capture_width
        self.cap_height = capture_height

        try:
            self.cap = cv2.VideoCapture(3)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)

            
            re, image = self.cap.read()
            if not re:
                raise RuntimeError('Could not read image from camera.')
            self.value = image
            self.start()
            print("succeed open cam")
        except:
            self.stop()
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(3)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        self.cap.release()

    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            cv2.imshow("preview", image)
            cv2.waitKey(5)
            if re:
                self.value = image
            else:
                break

    def save_snapshot(self, directory):
        image_path = os.path.join(directory, str(uuid1()) + '.jpg')
        cv2.imwrite(image_path, self.value)

    def save_free(self):
        self.save_snapshot(self.free_dir)
        self.free_count = len(os.listdir(self.free_dir))
        print('Free image saved.')

    def save_blocked(self):
        self.save_snapshot(self.blocked_dir)
        self.blocked_count = len(os.listdir(self.blocked_dir))
        print('Blocked image saved.')

    @staticmethod
    def instance(*args, **kwargs):
        return DataCollection(*args, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--free', action='store_true')
    parser.add_argument('--blocked', action='store_true')
    args = parser.parse_args()

    dc = DataCollection()
    if args.free:
        dc.save_free()
    if args.blocked:
        dc.save_blocked()
    if not args.free and not args.blocked:
        print("Specify '--free' or '--blocked'.")
    dc.stop()
