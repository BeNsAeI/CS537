import cv2
import numpy as np
threshold = 0.516
filename = './images/NotreDame5.jpg'
img = cv2.imread(filename)
blur = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,128,256)
cv2.imshow('Edge',edges)
edges = np.float32(edges)
harris = cv2.cornerHarris(edges,2,3,0.04)

#result is dilated for marking the corners, not important
Output = cv2.dilate(harris,None)
print(len(Output[Output > threshold*Output.max()]))
# Threshold for an optimal value, it may vary depending on the image.
img[Output>threshold*Output.max()]=[255,0,0]
cv2.imshow('Output',img)
if cv2.waitKey(0) & 0xff == 27:
   cv2.destroyAllWindows()
