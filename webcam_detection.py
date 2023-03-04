# import libraries
import cv2
import numpy as np

yolo = cv2.dnn.readNet('./yolov5_v2_new.onnx')

# object detection function
def imgRead(img):
    height, width, _ = img.shape
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image = img, scalefactor = 1 / 255, size = (640, 640), mean = (0, 0, 0), swapRB=True, crop=False)
    # blob oject is given as input to the network
    yolo.setInput(blob)
    # get the index of the output layers
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    # forward pass
    layeroutput = yolo.forward(output_layer_name)

    # post processing
    boxes = []
    confidences = []
    
    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            print(score)
            class_id = np.argmax(score)
            confidence = score[class_id]
            
            if confidence > 0.05:
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # find top left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                
                # get indexes of the object(s) detcted after supressing redundant bounding boxes
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                # assign font
                font = cv2.FONT_HERSHEY_COMPLEX
                # assign colors to the bounding boxes
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                
                # add bounding boxes to each object in the image frame
                try:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        color = colors[i]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.imshow('img', img)

                except:
                    pass

# capture the webcam feed
cap = cv2.VideoCapture(0)

while(True):
    # read the camera frame
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    out = imgRead(frame)

    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()