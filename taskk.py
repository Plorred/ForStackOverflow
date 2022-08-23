from imutils.video import FPS
import numpy as np
import argparse
import cv2
import time
import dlib

from multiprocessing import Value

# here must an argument for the second camera(this case second video) but I didnt do this since I dont know how to deal with my task
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", type=int, required=True,
	choices=[0, 1, 2], default = 0)
ap.add_argument("-i", "--input", type=str,
	help="path to the input video file")
args = vars(ap.parse_args())


#SSD detection each 15 frames
skip_frames  = 15


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
	
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	"mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

net.setPreferableTarget  (cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend (cv2.dnn.DNN_BACKEND_OPENCV)

dnnWork = 0


if not args.get("input", False):
	print("[INFO] : Camera streaming")
	vs = cv2.VideoCapture('v4l2src ! video/x-raw,width=640,height=480 ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

	# warmup for webcam
	time.sleep(2.0)
else:
	print("[INFO] : Opening input...")
	vs = cv2.VideoCapture(args["input"])

fps = FPS().start()

W = None
H = None


# frames counter
noFrames    = 0
confidence  = 0

mainHR_v      = Value('i', 1)

# here i tried to implement my task, I created a dictionary which will contain every unique person with its id, which increases with every person
id = 0
detected_persons = {}





while mainHR_v.value:
	# grabbing next frame
	if not args.get("input", False):
		# вебка
		ret, frame = vs.read()

		if ret == False:
			print ("[ERROR] Can't get a frame")
		else:
			frame = cv2.flip(frame,0)   
	else: 
		frame = vs.read()
		frame = frame[1]

	# video ends
	if frame is None:
		break
		
	# convertation from BGR to RGB
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)		


	# Initializing width and height
	if W is None or H is None:
		(H, W) = frame.shape[:2]				

	if noFrames % skip_frames == 0:
		dnnWork = 1

		# here we will be saving trackers and their confidences
		trackers = []
		confidences = []

		# preparing frame for SSD
		blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
		# print("First Blob: {}".format(blob.shape))
	
		# Inputting a frame into SSD
		net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])	
		networkOutput = net.forward()


		for detection in networkOutput[0, 0]:
			
			humanClass = int(detection[1])
			if CLASSES[humanClass] != "person":
				continue
			id += 1
			confidence = float(detection[2])
		
			if confidence > 0.35:
				
				confidences.append(confidence)
				
				left   = int(detection[3]*W)
				top    = int(detection[4]*H)
				right  = int(detection[5]*W)
				bottom = int(detection[6]*H)
				
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
				
				###########################################################################
				#here im trying to add to dictionary every unique person but how do I tell this second camera(at this case second video) idk 
				
				detected_persons[f"id: {id}"] = {
						"id": id,
						"bbox": (left, top, right, bottom),
						"confidence": confidence*100,
						"color": np.random.randint(0, 255, size=3),
					}



				###########################################################################
				label = "{}: {:.2f}%".format("Confidence", confidence*100)
				y = top - 15 if top - 15 > 15 else top + 15
				cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				
				# box into tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(left, top, right, bottom)
				tracker.start_track(rgb, rect)

				# saving tracker
				trackers.append(tracker)
		#after end of ssd detection I initialize id with zero
		id = 0
				


								
	else:
		dnnWork = 0
		i = 0
		for tracker in trackers:
			tracker.update(rgb)
			pos = tracker.get_position()

			left   = int(pos.left())
			top    = int(pos.top())
			right  = int(pos.right())
			bottom = int(pos.bottom())
			
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
				
			label = "{}: {:.2f}%".format("Confidence", confidences[i]*100)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			
			i +=1
	#dont think about mode arguments stuff	
	if args["mode"] == 0:
			cv2.imshow("Frame", frame)
			#video is playing toooooo fast so im slowing down the whole process
			time.sleep(0.04)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	noFrames += 1
	fps.update()

# program duration and avg fps
fps.stop()
print("[INFO] : Duration: {:.2f}".format(fps.elapsed()))
print("[INFO] : Avg. FPS:  {:.2f}".format(fps.fps()))


vs.release()
print("[INFO] : Closing all windows")
cv2.destroyAllWindows()
mainHR_v.value = 0
print(detected_persons)
print("[INFO] : END")



