from  ultralytics import YOLO
import cv2 as cv
from Pose_module import detector

vid=cv.VideoCapture("benchmark-sample2.mp4")

model=YOLO("best.pt")
ps=detector()
while vid.isOpened():
    isTrue,frame=vid.read()
    mp=ps.find_pose(frame)
    results = model(mp)
    plotted_frame = results[0].plot()
    for box in results[0].boxes:
        print(box)
    cv.imshow("result", plotted_frame)
    cv.waitKey(10)
