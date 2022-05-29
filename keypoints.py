import sys
import os
import dlib
import cv2
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
win.set_title("Face Landmarks Detection")

video_path = "sx304-video-mrjo0.avi"
video = cv2.VideoCapture(video_path)

frames_count = 0
landmarks_flow = [[] for _ in range(68)]

while True:
	success, image = video.read()
	if not success:
		break

	frames_count += 1

	# open image with cv2
	#image = cv2.imread(image_path)

	# resize image
	width = 500
	height = int(width / image.shape[1] * image.shape[0])
	dsize = (width, height)

	image = cv2.resize(image, dsize)

	# grayscale image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# draw image
	win.set_image(image)

	# detect faces in image
	faces = detector(image, 1)
	#print("Number of faces detected: {}".format(len(faces)))

	# TODO: What to do with multiple faces in 1 frame?
	if len(faces) == 0:
		print("Could not detect face")
		break

	face = faces[0]

	#print("Face: Left: {} Top: {} Right: {} Bottom: {}".format(face.left(), face.top(), face.right(), face.bottom()))

	# Get the landmarks/parts for the face
	landmarks = predictor(image, face)
	#print("Part 0: {}, Part 1: {} ...".format(landmarks.part(0), landmarks.part(1)))

	# TODO: What to do if the 68 landmarks are not detected?
	if landmarks.num_parts != 68:
		print("Could not detect landmarks")
		break
		
	# draw landmarks
	win.clear_overlay()
	for i, part in enumerate(landmarks.parts()):
		landmarks_flow[i].append([part.x, part.y])
		win.add_overlay_circle(part, 2)

# convert to numpy array
landmarks_flow = np.array(landmarks_flow)

for i, landmark_flow in enumerate(landmarks_flow):
	distances = [np.linalg.norm(landmark_flow[i] - landmark_flow[i-1]) for i in range(1, frames_count)]

	print("Landmark #{:02d}: Min {:.2f} Max {:.2f} Avg {:.2f} Std {:.2f}".format(
		i + 1,
		np.amin(distances),
		np.amax(distances),
		np.average(distances),
		np.std(distances)
	))

dlib.hit_enter_to_continue()