from darkflow.net.build import TFNet
import os
print('working' + os.getcwd())


options = {"model": "/home/falcon01/PycharmProjects/Track_it/darkflow/cfg/yolo_custom.cfg",
           "load": "/home/falcon01/PycharmProjects/Track_it/darkflow/bin/yolo.weights",
           "batch": 1,
           "gpu": 0.1,
           "epoch": 4,
           "train": True,
           "annotation": "/home/falcon01/Desktop/Dataset/annotations/",
           "dataset": "/home/falcon01/Desktop/Dataset/images/"}
tfnet = TFNet(options)
tfnet.train()
tfnet.savepb()
