import cv2 
import numpy as np 

#1. initialize the image using numpy array
#create a blank white image ( 500*500 pixels, 3 channels for RGB)

image=np.ones((500,500,3), dtype=np.uint8)*255

#2. draw unfilled and filled square (rectangles)
# unfilled square - blue color, thickness 3
cv2.rectangle(image, (50,50), (150,150), (255,0,0), 3)

#Filled square- green color, thickness -1 (filled)
cv2.rectangle(image, (200,50), (300, 150), (0,255,0), -1)

#3. Draw unfilled and filled circle 
#Unfilled circle - red color, center (100,250), radius 50, thickness 3
cv2.circle(image, (100, 250), 50, (0,0,255), 3)

#Filled circle - yellow color, center (250,250), radius 50, thickness -1 (filled)
cv2.circle(image, (250,250), 50, (0,255,255), -1)

#4. Draw a line
#Line from (50, 350) to (450, 350), magenta color, thickness 5
cv2.line(image, (50,350), (450,350), (255,0,255),5)

#5. Write text on the image
#Text properties
font=cv2.FONT_HERSHEY_SIMPLEX
text= "OpenCV Drawing"
position =(80,450)
font_scale=1
color=(0,0,0) #Black color
thickness=2 

cv2.putText(image, text, position, font, font_scale, color, thickness)

#Display the image
cv2.imshow('Drawing with OpenCV', image)

#Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

#Optiona: Save the image
cv2.imwrite('opencv_drawing_output.png', image)