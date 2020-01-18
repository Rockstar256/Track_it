from collections import namedtuple

import xml.etree.ElementTree as ET
from darkflow.net.build import TFNet
import cv2
from glob import glob
import os


def predicted_rectangle(predictions):
    max = 0
    index = 0
    for i, result in enumerate(predictions):
        if result['confidence'] > max:
            max = result['confidence']
            index = i
    predicted = predictions[index]
    if predicted['confidence'] <= 0.6:
        return None
    top_x = predicted['topleft']['x']
    top_y = predicted['topleft']['y']

    btm_x = predicted['bottomright']['x']
    btm_y = predicted['bottomright']['y']

    confidence = predicted['confidence']
    label = str(round(confidence, 3))
    # print(label)
    pred_rect = namedtuple('pred_rect', 'top_x top_y btm_x btm_y')
    return pred_rect(top_x, top_y, btm_x, btm_y)


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.btm_x, b.btm_x) - max(a.top_x, b.top_x)
    dy = min(a.btm_y, b.btm_y) - max(a.top_y, b.top_y)
    if (dx >= 0) and (dy >= 0):
        return dx*dy


options = {"model": "cfg\\yolo_custom.cfg",
           "load": -1,
           "gpu": 0.1}
tfnet2 = TFNet(options)
tfnet2.load_from_ckpt()
# Change the path according to the windows directory
# Don't forget to keep \\ as separator in windows path
# See whether all images are of jpg extension
# Keep images and annotations in separate folders
test_image_paths = glob(os.path.join('/home/falcon01/Desktop/test/images/', '*.jpg'))
for image_path in test_image_paths:
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet2.return_predict(original_img)
    image_name = os.path.basename(image_path)
    xml_file_name = image_name[:-4] + '.xml'
    # Change the path according to requirement
    xml_file = open(os.path.join(os.path.dirname(image_path), '../annotations/{}'.format(xml_file_name)))


    pred_rectcoord = predicted_rectangle(results)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        xml_box = obj.find('bndbox')
        xmin = int(float(xml_box.find('xmin').text))
        ymin = int(float(xml_box.find('ymin').text))
        xmax = int(float(xml_box.find('xmax').text))
        ymax = int(float(xml_box.find('ymax').text))
    xml_file.close()
    org_rect = namedtuple('original_box', 'top_x top_y btm_x btm_y')
    org_rect_coord = org_rect(xmin, ymin, xmax, ymax)
    df_x = org_rect_coord.btm_x - org_rect_coord.top_x
    df_y = org_rect_coord.btm_y - org_rect_coord.top_y
    org_area = df_x * df_y
    print('Area of the original box is :', org_area)
    intersected_area = area(pred_rectcoord, org_rect_coord)
    if not intersected_area:
        with open('truth_or_dare.txt', mode='at', encoding='utf-8') as f:
            f.write('0\n')
        continue
    part_covered = float(intersected_area)/int(org_area)
    with open('truth_or_dare.txt', mode='at', encoding='utf-8') as f:
        if part_covered >= 0.5:
            f.write('1\n')
        else:
            f.write('0\n')

with open('truth_or_dare.txt', mode='rt', encoding='utf-8') as f:
    count_true = 0
    count_false = 0
    for line in f.readlines():
        value = int(line.strip())
        if value == 0:
            count_false += 1
        else:
            count_true += 1

print('The test accuracy is :', float(count_true)/(count_true + count_false))
# If you want to run again this program you should delete the truth or dare text file.
