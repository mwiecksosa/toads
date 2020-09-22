import random
import os
import time
import cv2 as cv
import numpy as np
import sys
import json
import pickle


#paths
darknetpath = "/u/erdos/students/mwiecksosa/darknet_TOAD/"
backuppath = darknetpath + "backup/"
datapath = darknetpath + "data/"
objpath = datapath + "obj/"
all_imgnames = datapath + "the_data_names.txt"
train = datapath + "train.txt"
test = datapath + "test.txt"
cfgpath = darknetpath+"cfg/"
origbackuppath = darknetpath + "/backup_weights_original/"

#initialize parameters
confThreshold = 0.3  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold
inpWidth = 640       #Width of network's input image
inpHeight = 480      #Height of network's input image

# Load names of classes
classesFile = datapath+"obj.names";
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = cfgpath + "yolo-obj.cfg";
modelWeights = backuppath + "yolo-obj_2000.weights";



def main():
	""" YOLO v3 custom cross-validation on cluster GPUs """

	with open(all_imgnames) as f:
	    content = f.readlines()

	# get full data, .jpg names
	full_raw_data = [objpath + x.strip('\n') for x in content]


	metrics_dict = dict()

	for split_num in range(30): # 30 train test splits

		sample_size = 200

		test_split = [full_raw_data[i] for i in sorted(random.sample(range(len(full_raw_data)), sample_size))]

		train_split = set(full_raw_data) - set(test_split)



		#clear data/obj/train.txt
		open(train, 'w').close()

		#write to data/obj/train.txt
		with open(train, 'a') as train_file:
			for name in train_split:
				train_file.write(name+'\n')

		#clear data/obj/train.txt
		open(test, 'w').close()
		#write to data/obj/train.txt
		with open(test, 'a') as test_file:
			for name in test_split:
				test_file.write(name+'\n')

		start = time.time()

		os.system('./darknet detector train data/obj.data cfg/yolo-obj.cfg darknet53.conv.74')

		while len(os.listdir(backuppath)) < 3: #1000 weights, 2000 weights, current weights
			continue

		stop = time.time()

		hours_elapsed = (stop - start) / 3600

		print("training took %f hours for split %i"%(hours_elapsed, split_num))

		#### evaluate model on test set
		#### using weights in backup/ folder, this is where they are saved
		TP_total,FP_total,TN_total,FN_total = evaluate_model(test_split,split_num)


		#### move weights, AFTER using them...
		new_weights_dir = darknetpath + "weights_split_" + str(split_num) + "/"

		if not os.path.exists(new_weights_dir):
		    os.makedirs(new_weights_dir)

		for weight_fname in os.listdir(backuppath):
			os.rename(backuppath+weight_fname,new_weights_dir+"split_num_"+str(split_num)+"_"+weight_fname)

		os.system("cp" + " " + train + " " + new_weights_dir + "train_" + str(split_num) + ".txt")

		os.system("cp" + " " + test + " " + new_weights_dir + "test_" + str(split_num) + ".txt")



		split_name = "split_"+str(split_num)
		metrics_dict[split_name] = (TP_total,FP_total,TN_total,FN_total)
		print("metrics dict:",metrics_dict)


	with open(darknetpath + "metrics_dict" + '.pkl', 'wb') as f:
		pickle.dump(metrics_dict, f, pickle.HIGHEST_PROTOCOL)



def evaluate_model(test_split,split_num):

	TP_total,FP_total,TN_total,FN_total = 0,0,0,0

	net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)



	datalist = []

	for img_name in test_split:
		datalist.append((objpath+img_name,objpath+img_name[:-4]+".txt"))

	counter = 0
	print("Split #%i:"%split_num,"%i predictions to make"%len(datalist))

	for data in datalist:
		counter += 1
		image_to_read = data[0]
		txtfile = data[1]

		frame = cv.imread(image_to_read)
		#cv.imshow("before pred",frame)
		#cv.waitKey(3000)

		# Create a 4D blob from a frame.
		blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

		# Sets the input to the network
		net.setInput(blob)

		# Runs the forward pass to get output of the output layers
		outs = net.forward(getOutputsNames(net))

		TP,FP,TN,FN = calc_TP_FP_TN_FN(frame, txtfile, outs)

		TP_total += TP
		FP_total += FP
		TN_total += TN
		FN_total += FN

		print("Split #%i"%split_num,"Pred #%i:"%counter,"TP_total = %i"%TP_total,"FP_total = %i"%FP_total,"TN_total = %i"%TN_total,"FN_total = %i"%FN_total)

		if counter is 10:
			break

		postprocess(frame, outs)

		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
		cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

		#cv.imshow("preds",frame.astype(np.uint8))
		#cv.waitKey(1000)

		cv.destroyAllWindows()

	return TP_total,FP_total,TN_total,FN_total


# Get the names of the output layers
def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]




def calc_TP_FP_TN_FN(frame, txt, outs):


	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]
	tp,fp,tn,fn = 0,0,0,0
	classIds = []
	confidences = []
	boxes = []


	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]


			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)


				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])


	indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)


	if any(map(len,indices)): #checks if the tuple is empty, if non-empty, has len ==> True, made predictions
		for i in indices:
			i = i[0]
			box = boxes[i]
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]

			drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
			center_x = int(left + width / 2)
			center_y = int(top + height / 2)


			x1_pred = int(center_x - width / 2)
			y1_pred = int(center_y - height / 2)
			x2_pred = int(center_x + width / 2)
			y2_pred = int(center_y + height / 2)


			if not os.stat(txt).st_size is 0: # if toad in the frame
				textfile = open(txt, "r")
				data = textfile.read().split(' ')

				x_true,y_true,w_true,h_true = float(data[1]), float(data[2]),float(data[3]),float(data[4].split("\n")[0])

				x1_true = int((x_true - w_true/2)*640) #absolute left bound
				x2_true = int((x_true + w_true/2)*640) #absolute right bound
				y1_true = int((y_true - h_true/2)*480) #absolute upper bound
				y2_true = int((y_true + h_true/2)*480) #absolute lower bound


				if abs(x1_true - x1_pred) > 25 and abs(x2_true - x1_pred) > 25 and abs(y1_true - y1_pred) > 25 and abs(y2_true - y2_pred) > 25: # if predictions are wildly off
					fp += 1

				else: #if pred values within 20 pixels of true values

					tp += 1


			else: # if size = 0, then no toad in the frame, and there are predictions, then fp


				fp += 1


	else: #if Tuple is empty ==> no predictions

		if not os.stat(txt).st_size is 0: #not empty ==> there is a toad in the image

			fn += 1 #prediction wouldn't be made, when there was a toad to predict

		else: #if file empty, then no toad, and no confident prediction, so tn += 1

			tn += 1



	return tp,fp,tn,fn





def drawPred(frame, classId, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

	label = '%.2f' % conf

	# Get the label for the class name and its confidence
	if classes:
		assert(classId < len(classes))
		label = '%s:%s' % (classes[classId], label)

	#Display the label at the top of the bounding box
	labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))



# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	classIds = []
	confidences = []
	boxes = []
	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
	indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)




if __name__ == '__main__':
	main()
