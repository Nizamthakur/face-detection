import cv2
from cv2 import cvtColor

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")# including haarcascadefiles 

img = cv2.imread("news.jpg", 1)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting rgb color to gray

faces = face_cascade.detectMultiScale(gray_img, scaleFactor= 1.1 , minNeighbors = 3)
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,100,0), 3)
print(faces)


cv2.imshow('gray', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

