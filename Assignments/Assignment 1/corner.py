import cv2
import numpy as np

filename = './images/NotreDame1.jpg'
img = cv2.imread(filename)
blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('Blurred',blur)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale',gray)
gray = np.float32(gray)
harris = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
Output = cv2.dilate(harris,None)

# Threshold for an optimal value, it may vary depending on the image.
img[Output>0.4214999999999813*Output.max()]=[255,0,0]
cv2.imshow('Output',img)
if cv2.waitKey(0) & 0xff == 27:
   cv2.destroyAllWindows()
