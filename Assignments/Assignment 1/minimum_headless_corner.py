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
step = 0.00025
def callibrate(Output):
    threshold = global_threshold
    count = 0
    coords = []
    while count > 210 or count < 200:
        count = len(Output[Output > threshold*Output.max()])
        if count > global_count:
            threshold += step
        elif count < global_count:
            threshold -= step
    print("Threshold found: " + str(threshold) + ", count is: " + str(count))
    for i in range(0,len(Output)):
        print ("-")
        for j in range(0,len(Output[i])):
            if (Output[i][j]>threshold*Output.max()):
                print("appending " + str([i,j]))
                coords.append([i,j])
    return coords
def main():
    for i in range (1, 6):
        path = './' + folder + '/' + filename + str(i) + '.' + extention
        print("Processing " + path + " ...")
        img = cv2.imread(path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        harris = cv2.cornerHarris(gray,2,3,0.04)
        Output = cv2.dilate(harris,None)
        coords = callibrate(Output)
        file = open('Points-' + str(i) +'.txt', 'w')
        for i in coords:
            string = "(" + str(i[0]) + ", " + str(i[1]) + ")\n"
            file.write(string)
            print(string)
        file.close()
if __name__ == '__main__': main()
