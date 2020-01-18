import matplotlib.pyplot as plt
import numpy as np
from darkflow.net.build import TFNet
import cv2 as cv


def boxing(original_img, predictions):
    with open('track.txt', mode='a', encoding='utf-8') as f:
        newImage = np.copy(original_img)

        max = 0
        index = 0
        for i, result in enumerate(predictions):
            if result['confidence'] > max:
                max = result['confidence']
                index = i
        predicted = predictions[index]
        if predicted['confidence'] <= 0.2:
            f.write('stop\n')
            return newImage
        top_x = predicted['topleft']['x']
        top_y = predicted['topleft']['y']

        btm_x = predicted['bottomright']['x']
        btm_y = predicted['bottomright']['y']

        mid_x = (top_x + btm_x)//2
        mid_y = (top_y + btm_y)//2

        f.write('{} {}'.format(mid_x, mid_y))

        confidence = predicted['confidence']
        label = str(round(confidence, 3))
        print(label)

        newImage = cv.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
        req_image = cv.putText(newImage, label, (top_x, top_y - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                               (0, 230, 0), 1, cv.LINE_AA)

    return req_image


video_path = '/home/falcon01/Desktop/football2.mp4'
options = {"model": "cfg/yolo_custom.cfg",
           "load": -1,
           "gpu": 0.1}
tfnet2 = TFNet(options)
tfnet2.load_from_ckpt()

# Video Processing
video = cv.VideoCapture(video_path)
window = cv.namedWindow('Processed')
print(video.isOpened())
while video.isOpened():
    ret, frame = video.read()
    # rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = tfnet2.return_predict(frame)
    cv.imshow('Processed', boxing(frame, results))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


