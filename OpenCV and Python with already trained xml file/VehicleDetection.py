import cv2
#our video
video = cv2.VideoCapture('Tesla Autopilot Dashcam Video.mp4')

#Our Pre-defined car classifier
classifier_file = 'car_detector.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

while True:

    #read the current frame
    (read_successful, frame) = video.read()

    #safe coding
    if read_successful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        

    #Display the image with the cars spotted
    cv2.imshow("Arghya's Car Detector", frame)

    #Don't autoclose (Wait here in the code adn listen for a key press)
    key = cv2.waitKey(1)

    #stop if S key is pressed
    if key==83 or key==115:
        break
 
#release the video capture
video.release()


'''
#Our Image
img_file = 'Car Image.jpg'

#create OpenCV image
img = cv2.imread(img_file)

#convert to grayscale (required for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)



#Display the image with the cars spotted
cv2.imshow("Arghya's Car Detector", img)

#Don't autoclose (Wait here in the code adn listen for a key press)
cv2.waitKey()
'''

print("Code Completed")