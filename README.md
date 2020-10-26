# object-detection-using-yolo-and-opencv

   In this project I've used transfer learning to load the yolov3 pretrained model and apply it to images and videos to detect the common objects present in them. Yolo or you only look once algorithm used  here is a state of the art real time object detection algorithm trained on the coco dataset by the Darknet team. In this repo you will find a yolo_video_final.py which is used for real time object detection on given video or webcam stream. I have used arguement parser to get the input and output paths of the video and setting some of the optional parameters for fine tuning. Opencv is used to draw the bounding boxes over detections, display the frame and write the output video to disk.
   
   
Steps to setup the environment and run the code:-
1. Make sure the required libraries show in requirements.txt are installed 
2. Make a directory with files named input,output and model to store the input/output data and the model.cfg and model.weights file
3. Open your Command Prompt / Terminal and cd into the directory you stored all the project files
4. Activate the environment that will run ur script
5. Pass in the required parameters of input and output file names and the model directory 


eg. of passing the parameters in cli :-
(YOUR ENV) C:\Users\<NAME>\Desktop\projects\yolo>python yolo_video_final.py --choice upload_video --input_path video.mp4 --output_path out_video.mp4 --yolo yolo_file
   
Since the weights file of the yolov3 model is quite large(237mb) I cannot upload it to github and u can find it here: https://pjreddie.com/darknet/yolo/
	 
	 
You can find a brief introduction to yolo algorithm in this medium article here : https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088


The 80 different objects that can be detected by yolo are : https://github.com/pjreddie/darknet/blob/master/data/coco.names
