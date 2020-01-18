import xml.etree.ElementTree as ET
from glob import glob
import os


annotations_path = '/home/falcon01/Desktop/dataset/export'
annotations_files = glob(os.path.join(annotations_path, '*.xml'))
for file in annotations_files:
    with open(file) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for obj in root.iter('object'):
            if obj.find('name').text == 'soccer':
                obj.find('name').text = 'ball'
                print('changed.')
                tree.write(file)
            else:
                print('unchanged')

