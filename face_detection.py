import cv2

#to make a cascade classifier object which contains all the face  features
face_cascade=cv2.CascadeClassifier("C:\\Users\\priya\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
#arg is  path to xml file which contains face features on basis of which face is detected

#reads or processes our img into 3 rgb matrix
img=cv2.imread("multi_face.jpg",1)


#cvt-convert;this cvtcolor func converts our image to a specifed color;here we convert our image to gray
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detectMultiscale func of face_cascade provides us with the cordinates of the rectangle that surround the detected face
#scalefactor -specify by how much percent the image size is to be reduced
#minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it. This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=3)

#tells the type of variable faces which is a matirx consisting of x,y,w,h(x and y are cordinats of intial point;w is width and h is height of rectangle)
print(type(faces))

#prints the matrix [[x.y.w.h]]
print(faces)

#adding rectangular box around the detected faces
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)#first arg is image around which we waant rectangle;2nd are 3rd arg are the points of rectangle(diagonally opp corners);4 th arg is the rbg code for the color of border of rectangle;5th arg is the thickness of the line of rectangle
    
#to display the image with detected face
cv2.imshow("face_detected",img)
cv2.waitKey(0)
cv2.destroyAllWindows()