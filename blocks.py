import cv2
import numpy as np
from time import sleep
from tqdm import tqdm

def lucas_kanade_method(video_path):
	cap = cv2.VideoCapture(video_path)

	# Parameter für ShiTomasi Corner-Detection (Keypoints?)
	feature_params = dict( maxCorners = 1000,
						qualityLevel = 0.1,
						minDistance = 1,
						blockSize = 10 )

	# Parameter für Lucas Kanade
	lk_params = dict( winSize  = (15,15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	color = np.random.randint(0,255,(100,3))

	# Ecken im ersten Frame
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	mask = np.zeros_like(old_frame)

	firstFrame = []
	lastFrame = None
	totalDistances = []

	# Progress
	videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=videoLength)

	frameCount = 1
	blockCount = len(old_gray)

	avgDistFrame = []

	while(1):
		pbar.update(1)

		ret,frame = cap.read()
		# Kein weiteres Bild mehr
		if ret == False:
			# Totale Distanz
			for i in range(len(firstFrame)):
				totalDistances.append(np.linalg.norm(lastFrame[i]-firstFrame[i]))
			break
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Optischer Fluss hier
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
		good_new = p1[st==1]
		good_old = p0[st==1]

		# Frames für totale Distanz
		lastFrame = good_new
		if len(firstFrame) == 0:
			firstFrame = good_old

		# Euklidische Distanz eines Blockes zwischen zwei Frames
		blockDistances = []
		for i in range(len(good_new)):
			blockDist = np.linalg.norm(good_new[i]-good_old[i])
			blockDistances.append(blockDist)

		# Durchschnittliche Distanz pro Frame
		distSum = 0
		for x in blockDistances:
			distSum += x

		avgDistFrame.append(distSum / blockCount)

		# Punktverlauf (nur Visualisierung)
#		for i,(new,old) in enumerate(zip(good_new,good_old)):
#			a,b = new.ravel()
#			c,d = old.ravel()
#			mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
#			frame = cv2.circle(frame,(int(a),int(b)),10,color[i].tolist(),-1)
#		img = cv2.add(frame,mask)

		#cv2.imshow('Optischer Fluss',img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)

		frameCount += 1

	pbar.close()

	print("\n---------- Durchschnittliche euklidische Distanz ----------\n")
	avgDist = 0
	for i in range(len(avgDistFrame)):
		print("Frame " + str(i+1) + ": " + str(avgDistFrame[i]))
		avgDist += avgDistFrame[i]

	print("\n---------- Totale euklidische Distanz zwischen Anfang und Ende ----------\n")
	avgTotalDist = 0
	for i in range(len(totalDistances)):
		print("Block " + str(i+1) + ": " + str(totalDistances[i]))
		avgTotalDist += totalDistances[i]

	print("\nGesamtdurchschnitt der Distanz: " + str(avgDist/len(avgDistFrame)) + "\n")
	print("Gesamtdurchschnitt der totalen Distanz: " + str(avgTotalDist/len(totalDistances)) + "\n")

	#cv2.waitKey(0)
	cv2.destroyAllWindows()
	cap.release()

video_path = "sx304-video-mrjo0.avi"
lucas_kanade_method(video_path)