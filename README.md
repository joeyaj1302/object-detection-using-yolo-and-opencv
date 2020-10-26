# object-detection-using-yolo-and-opencv

   In this project I've used transfer learning to load the yolov3 pretrained model and apply it to images and videos to detect the common objects present in them. Yolo or you only look once algorithm used  here is a state of the art real time object detection algorithm trained on the coco dataset by the Darknet team. In this repo you will find a yolo_video_final.py which is used for real time object detection on given video or webcam stream. I have used arguement parser to get the input and output paths of the video and setting some of the optional parameters for fine tuning. Opencv is used to draw the bounding boxes over detections, display the frame and write the output video to disk.
	 
	 
You can find a brief introduction to yolo algorithm in this medium article here : https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
The 80 different objects that can be detected by yolo are : https://github.com/pjreddie/darknet/blob/master/data/coco.names
