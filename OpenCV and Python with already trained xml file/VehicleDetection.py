import cv2
#our video
video = cv2.VideoCapture('Video File Name.mp4')

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

print("Code Completed")
