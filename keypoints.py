import sys
import os
import dlib
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
win.set_title("Face Landmarks Detection")

video_path = "sx304-video-mrjo0.avi"
video = cv2.VideoCapture(video_path)

prev_landmarks = None

while True:
	success, image = video.read()
	if not success:
		break

	# open image with cv2
	#image = cv2.imread(image_path)

	# resize image
	width = 1000
	height = int(width / image.shape[1] * image.shape[0])
	dsize = (width, height)

	image = cv2.resize(image, dsize)

	# grayscale image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# draw image
	win.set_image(image)

	# detect faces in image
	faces = detector(image, 1)
	print("Number of faces detected: {}".format(len(faces)))

	# TODO: What to do with multiple faces in 1 frame?
	if len(faces) == 0:
		print("Could not detect face")
		break

	face = faces[0]

	print("Face: Left: {} Top: {} Right: {} Bottom: {}".format(
		face.left(), face.top(), face.right(), face.bottom()))

	# Get the landmarks/parts for the face
	landmarks = predictor(image, face)
	print("Part 0: {}, Part 1: {} ...".format(
		landmarks.part(0), landmarks.part(1)))

	# TODO: What to do if the 68 landmarks are not detected?
	if landmarks.num_parts != 68:
		print("Could not detect landmarks")
		break
		

	# draw landmarks
	win.clear_overlay()
	#for part in landmarks.parts():
		#win.add_overlay_circle(part, 3)

	if prev_landmarks:
		lines = [dlib.line(prev_part, part) for (prev_part, part) in zip(prev_landmarks.parts(), landmarks.parts())]
		
		for line in lines:
			win.add_overlay(line, dlib.rgb_pixel(255, 255, 255))

	# save landmarks
	prev_landmarks = landmarks

dlib.hit_enter_to_continue()