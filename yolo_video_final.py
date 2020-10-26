#Using transfer learning to import pretrained yolo model and its weights to predict/detect objects on video in real time

import cv2
import numpy as np
import imutils
import argparse
import os
import time

#contructing the arguement parser
#sample of using arguement parser :-
#1. Open CMD or terminal and cd into the project directory eg- C:\Users\Jithil\Desktop\projects\yolo
#2. Activate the python environment which has all the packages eg- if using anaconda env use conda activate <your env name>
#3  Run the arg parser or type --help to understand the arguements eg - python yolo_video_trial.py --choice webcam --output_path out_snapshot4.avi --yolo yolo-coco

ap = argparse.ArgumentParser()
ap.add_argument("-ch","--choice",required=True,help = "specify if you want to load the video or use your webcam")
ap.add_argument("-i","--input_path",required=False, help="specify the input path of the video")
ap.add_argument("-o","--output_path",required=True,help="specify the output path of the image")
ap.add_argument("-y","--yolo",required= True, help = "Give the path to the yolo directory where the yolo model and pretrained weights are stored")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.4,help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#get the class names from the coco.names folder 
Labels_path = os.path.sep.join([args['yolo'],'coco.names'])
LABELS = open(Labels_path).read().strip().split("\n")
print("The following objects can be detected in the video :")
print(" ,".join(x for x in LABELS))

#the paths to the model.cfg and model.weights i.e the weights file
weights_path = os.path.sep.join([args['yolo'],'yolov3.weights'])
config_path = os.path.sep.join([args['yolo'],'yolov3.cfg'])

np.random.seed(2)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")   #Different colour names for different objetcs to be detected

print("========================LOADING THE YOLO MODEL============================")
model = cv2.dnn.readNetFromDarknet(config_path,weights_path)
ln = model.getLayerNames()
ln = [ln[i[0]-1] for i in model.getUnconnectedOutLayers()]#As we do not want all the convolutional layers only the 3 layers where prediction takes place in a yolo model
print(ln)

#getting user choice of video from cli agparser
if args["choice"] == "upload_video":
    vid = cv2.VideoCapture(args['input_path'])
elif args['choice'] == "webcam":
    vid = cv2.VideoCapture(0)

#instantiating the writer object that will later help in writing the predicted output video to disk
writer = None
(W,H) = (None,None)

while True:
    (confirmed , frame) = vid.read() #getting frames from video stream
    if not confirmed:
        break
    if W is None and H is None:
        (H,W) = frame.shape[:2]
        print("====================Video processing started====================")
        print("The video resolution is :\n{}x{}".format(W,H))
        
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)  #BLob is similar to Image augmentation present in image data generators                                                                                         
    model.setInput(blob)                                                                 #of tensorflow
    start = time.time()
    layer_outputs = model.forward(ln)         #The captured frame is being passsed into the yolo model and the output is stored in layer_outputs
    end = time.time()

    #Declaring the lists that will contain data about class detections,co-ordinates of bounding boxes and the predicted classID's
    boxes = []
    confidences = []
    classIDs = []

    for outputs in layer_outputs:
        for detection in outputs:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence>args['confidence']:       #Confidence values can be tuned from the optional args to get desired outputs
                box = detection[0:4] * np.array([W, H, W, H])
                (centerx,centery,width,height) = box.astype('int')
                x = int(centerx - (width/2))         #getting the left most starting point from the center points for drawing bounding box later
                y = int(centery - (height/2))
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,args['confidence'],args['threshold']) #Non max supression to remove over lapping boxes

    #now to draw the bounding rectangles over the images
    #Make sure atleast one object is detected
    if len(indexes)>0:
        for i in indexes.flatten():
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            text = "{}: {}%".format(LABELS[classIDs[i]], int(confidences[i]*100))
            cv2.putText(frame,text,(x, y - 5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 2)
            cv2.imshow("video",frame)               #For displaying the frame being passed through the model and showing real time predictions
            if cv2.waitKey(1) & 0xFF == ord('q'):   #pressing the escape key breaks the loop for webcam video stream
                cv2.destroyAllWindows()
                break

    #Check if the video writer is None
    if writer is None:
        #Initialize our video writer to write the output video with predictions to output path specified on disk
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output_path"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)
    key = cv2.waitKey(1)
    if key == 27:
        print("...............Stopping Recording................")
        break
# release the file pointers
writer.release()
vid.release()
print("====================Done========================")
