'''
Object detection ("Ball tracking") with OpenCV
    Adapted from the original code developed by Adrian Rosebrock
    Visit original post: https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
Developed by Marcelo Rovai - MJRoBot.org @ 7Feb2018 
'''

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "yellow object"
# (or "ball") in the HSV color space, then initialize the
# list of tracked points
colorLower = (0, 0, 0)
colorUpper = (170, 250, 40)
colorLower2 = (0, 0, 0)
colorUpper2 = (170, 250, 120)
pts = deque(maxlen=args["buffer"])

 
# if a video path was not supplied, grab the reference
# to the webcam
#if not args.get("video", False):
	#camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
#else:
camera = cv2.VideoCapture(args["video"])

#make empty lists for coordinates
xcoords=[]
ycoords=[]
total_coords=[]


(hAvg, sAvg, vAvg) = (None, None, None)
total = 0
#################################################################################################################

##########################################################################################################
#camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	ytot = [1]
	xtot = [1]


	# grab the current frame
	(grabbed, frame) = camera.read()
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
 
	# resize the frame, inverted ("vertical flip" w/ 180degrees),
	# blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=800)
	frame = imutils.rotate(frame, angle=180)
	
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, colorLower, colorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	#####################################################################################################################

	#daniel'sstuff
	#mask3 = mask -mask2
	cv2.imshow("output", mask)


	cv2.waitKey(0)

	#############################################s###################################################################

	# display the image
	########################################################################################################################
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

 
		# only proceed if the radius meets a minimum size
		if radius > 1:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

 
	# update the points queue
	pts.appendleft(center)

	#*************NEW SECTION*************
	#name list for x,y coordinates 
	xcoords.append([center[0]])
	ycoords.append([center[1]])
	total_coords.append([center])

	
		# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
 
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	cv2.imshow('mask', mask)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

#put lists into matrix form
all_coords=np.array(total_coords)
x_coords=np.array(xcoords)
y_coords=np.array(ycoords)

#find displacements
tot_disp = all_coords-all_coords[0]
x_disp=x_coords-x_coords[0]
y_disp=y_coords-y_coords[0]

##########################################################################################################################
#removes drift
dx_disp = []
dx_disp.append(x_disp[0])
for i in range(1,len(x_disp)):
	dx_disp.append(x_disp[i]-x_disp[i-1])


#removes large spikes
mean = np.mean(dx_disp, axis=0)
sd = np.std(dx_disp, axis=0)

list = []
for x in dx_disp:
	if x > mean + sd or x < mean - sd:
		list.append(mean)
	else:
		list.append(x)

final_list = np.array(list)


######################################################################################################################

#find relative max
#import scipy
#from scipy.signal import find_peaks
#max_x=scipy.signal.find_peaks(x_disp,height=0,threshold=10)




#find fps
camera = cv2.VideoCapture(args["video"])
fps= camera.get(cv2.CAP_PROP_FPS)

#make time vector
#time=np.arange(1,len(x_disp)+1,1)
length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))#number of frames
time=np.arange(1,length+1,1)
time=time/fps #time in seconds
time=np.around(time, decimals=2)

#plot displacement over time
import matplotlib.pyplot as plt
#plt.plot(time,tot_disp)
#plt.ylabel('Total Displacement')
#plt.xlabel('Time (sec)')
#fig=plt.figure()
#fig.savefig('Total (x and y) Displacement')

fig=plt.figure()
plt.plot(time,x_disp)
plt.ylabel('Horizontal Displacement')
plt.xlabel('Time (sec)')
fig.savefig('Horizontal Displacement.pdf')

fig=plt.figure()
plt.plot(time,y_disp)
plt.ylabel('Vertical Displacement')
plt.xlabel('Time (sec)')
fig.savefig('Vertical Displacement.pdf')
camera.release()
###########################################################################################

fig=plt.figure()
plt.plot(time, final_list)
plt.ylabel('Horizontal Displacement')
plt.xlabel('Time (sec)')
fig.savefig('Daniel_Horizontal Displacement.pdf')
