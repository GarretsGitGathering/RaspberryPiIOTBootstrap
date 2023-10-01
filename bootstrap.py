# Imports
import mediapipe as mp
from picamera2 import Picamera2
import numpy as np
import time
import cv2


# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

# use mediapipe to draw detect and draw figure
def draw_pose(image, landmarks):  

	# copy the image
	landmark_image = image.copy()
	
	# get the dimensions of the image
	height, width, _ = image.shape
 
	# create landmarks list
	pose_landmarks_list = landmarks.pose_landmarks
  
	#iterate through landmarks list
	for landmark in range(len(pose_landmarks_list.landmark)):
  
		# draw landmark points on image  
		mp.solutions.drawing_utils.draw_landmarks(
			landmark_image,
			pose_landmarks_list,
			mp.solutions.pose.POSE_CONNECTIONS,
			mp.solutions.drawing_styles.get_default_pose_landmarks_style())
	return landmark_image

def main():
	# continuously loop
	while True:
		# set image to image from camera
		image = pi_camera.capture_array()

		# Create a pose estimation model 
		mp_pose = mp.solutions.pose
		
		# start detecting the poses
		with mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as pose:

			# To improve performance, optionally mark the image as not 
			# writeable to pass by reference.
			image.flags.writeable = False
			
			# get the landmarks
			results = pose.process(image)
			
			# detect landmarks and set image accordingly 
			if results.pose_landmarks != None:
				result_image = draw_pose(image, results)
				cv2.imwrite('output.png', result_image)
				print(results.pose_landmarks)
			else:
				# set result_image to image to avoid gaps in video
				result_image = image
				print('No Pose Detected')
    
			# display image
			cv2.imshow("Video", result_image)
   
			# This waits for 1 ms and if the 'q' key is pressed it breaks the loop	 
			if cv2.waitKey(1) == ord('q'):
				break

	# Close all the windows
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
	print('done')
