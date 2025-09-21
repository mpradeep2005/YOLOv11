from  ultralytics import YOLO
import cv2 as cv

img=cv.imread("pexels-freestockpro-1031698.jpg")
img=cv.resize(img,(640,480))

model=YOLO("yolo11n.pt")
results = model(img)

plotted_frame=results[0].plot()

for box in results[0].boxes:
    print(box)



cv.imshow("result",plotted_frame)
cv.waitKey()
