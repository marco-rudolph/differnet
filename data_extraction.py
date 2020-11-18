import cv2
from xml.dom import minidom
import config as c

# Load videos one by one
for i in range(c.num_videos):
    print('Data generation on video-' + str(i+1))
    filename = str(i+1)

    # Opens the Video file
    cap = cv2.VideoCapture('dataset/data-generation/videos/' + filename + '.avi')

    # Read annotations
    annotation = minidom.parse('dataset/data-generation/annotations/' + filename + '.xml')
    boxes = annotation.getElementsByTagName('box')

    frameList = []
    labelList = []
    boxesList = []

    # Store the bounding box info along with frame number info into list
    for i in range(boxes.length):

        # make sure not select the bounding box that outside the frame
        if (boxes[i].attributes['outside'].value != '1'):
            frame = int(boxes[i].attributes['frame'].value)
            frameList.append(frame)

            labelList.append(boxes[i].parentNode.attributes['label'].value)

            ytl = int(float(boxes[i].attributes['ytl'].value))
            ybr = int(float(boxes[i].attributes['ybr'].value))
            xtl = int(float(boxes[i].attributes['xtl'].value))
            xbr = int(float(boxes[i].attributes['xbr'].value))
            boxesList.append([ytl, ybr, xtl, xbr])

    # Set up shrink percentage
    shrink_percentage = 0.02
    j = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frame is not None and j in frameList):
            ytl = boxesList[frameList.index(j)][0]
            ybr = boxesList[frameList.index(j)][1]
            xtl = boxesList[frameList.index(j)][2]
            xbr = boxesList[frameList.index(j)][3]
            label = 'good' if labelList[frameList.index(j)] == 'bottle' else 'defect'

            # Crop the frames with the bounding box position info
            crop_frame = frame[int(ytl*(1+shrink_percentage)):int(ybr*(1-shrink_percentage)),
                         int(xtl*(1+shrink_percentage)):int(xbr*(1-shrink_percentage))]

            # output file formatting example "video1-frame4-defect.jpg"
            print('Successfully generated: ' +c.save_cropped_image_to + label + '/video-' + filename + '-frame' + str(j) +
                  '-' + label + '.jpg')
            cv2.imwrite(c.save_original_image_to + label + '/original-video-' + filename + '-frame' + str(j) + '-' + label + '.jpg',
                        frame)
            cv2.imwrite(c.save_cropped_image_to + label + '/video-' + filename + '-frame' + str(j) + '-' + label + '.jpg',
                        crop_frame)

        if ret == False:
            break
        j += 1

    cap.release()
    cv2.destroyAllWindows()
