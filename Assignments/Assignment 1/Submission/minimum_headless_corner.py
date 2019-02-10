import cv2
import numpy as np
#from numba import vectorize
#vectorize(['float32(float32, float32)'], target='cuda')
folder = 'images'
filename = 'NotreDame'
extention = 'jpg'
#global_threshold = 0.00125
global_threshold = 0.001
global_count = 200
step = 0.0025
coord_tolerance = 10
torch_limit = 1200
def callibrate(Output):
    threshold = global_threshold
    count = 0
    coords = []
    while count > 4000 or count < 300:
        count = len(Output[Output > threshold*Output.max()])
        if count > global_count:
            threshold += step
        elif count < global_count:
            threshold -= step
        print(threshold)
    print("Threshold found: " + str(threshold) + ", count is: " + str(count))
    index = np.where(Output>threshold*Output.max())
    coords = zip(index[0], index[1])
    #print(coords)
    #print(len(coords))
    return coords
def main():
    for i in range (1, 6):
        i = 4
        path = './' + folder + '/' + filename + str(i) + '.' + extention
        print("Processing " + path + " ...")
        img = cv2.imread(path)
        blur = cv2.GaussianBlur(img,(5,5),0)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,128,256)
        edges = np.float32(edges)
        harris = cv2.cornerHarris(edges,2,3,0.04)
        Output = cv2.dilate(harris,None)
        coords = callibrate(Output)
        file = open('Points-' + str(i) +'.txt', 'w')
        count = 1
        x = 0
        y = 0
        for i in coords:
            if (i[0] < torch_limit and i[1] < torch_limit) and ((i[0] > x + coord_tolerance or i[0] < x - coord_tolerance) or (i[1] > y + coord_tolerance or i[1] < y - coord_tolerance)):
                x = i[0]
                y = i[1]
                string = "(" + str(i[0]) + ", " + str(i[1]) + "),"
                file.write(string)
                count += 1
            if count > 200:
                break
        print(count)
        file.close()
        break
if __name__ == '__main__': main()
