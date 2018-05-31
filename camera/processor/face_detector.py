from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np
import cv2

class FaceDetector(object):
    def __init__(self, flip = True):
        self.vs = PiVideoStream(resolution=(800, 608)).start()
        self.flip = flip
        time.sleep(2.0)

        # opencvの顔分類器(CascadeClassifier)をインスタンス化する

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        frame = self.process_image(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def process_image(self, frame):
        # opencvでframe(カラー画像)をグレースケールに変換
        img = cv2.imread(frame)
        
        # 上記でグレースケールに変換したものをインスタンス化した顔分類器の
        # detectMultiScaleメソッドで処理し、認識した顔の座標情報を取得する
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        # 取得した座標情報を元に、cv2.rectangleを使ってframe上に
        # 顔の位置を描画する
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        
        # frameを戻り値として返す
        return frame
