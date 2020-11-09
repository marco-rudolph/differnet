import cv2
from xml.dom import minidom

# Read annotations
annotation = minidom.parse('dataset/data-generation/annotations.xml')
boxes = annotation.getElementsByTagName('box')
boxesList = []
frameList = []

for i in range(boxes.length):
    frame = int(boxes[i].attributes['frame'].value)
    frameList.append(frame)

    ytl = int(float(boxes[i].attributes['ytl'].value))
    ybr = int(float(boxes[i].attributes['ybr'].value))
    xtl = int(float(boxes[i].attributes['xtl'].value))
    xbr = int(float(boxes[i].attributes['xbr'].value))

    boxesList.append([ytl, ybr, xtl, xbr])


# one specific item attribute
print('Box #1 frame:')
print(boxes[0].attributes['frame'].value)

# Opens the Video file
cap = cv2.VideoCapture('dataset/data-generation/FileOutput0_2019-07-06_17-11-17-01-01 black jar short.avi')
shrink_percentage = 0.02
j = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if(frame is not None and j in frameList):
        ytl = boxesList[frameList.index(j)][0]
        ybr = boxesList[frameList.index(j)][1]
        xtl = boxesList[frameList.index(j)][2]
        xbr = boxesList[frameList.index(j)][3]

        crop_frame = frame[int(ytl*(1+shrink_percentage)):int(ybr*(1-shrink_percentage)),
                     int(xtl*(1+shrink_percentage)):int(xbr*(1-shrink_percentage))]
        print('frame', j)
        cv2.imwrite('dataset/data-generation/frame' + str(j) + '.jpg', crop_frame)

    if ret == False:
        break
    j += 1

cap.release()
cv2.destroyAllWindows()
