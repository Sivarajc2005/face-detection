import cv2
alg="haarcascade_frontalface_default.xml"
lod_alg=cv2.CascadeClassifier(alg) #loading algrorithm 
cam=cv2.VideoCapture(0) #initializing camera
while True:
    _,img=cam.read()  #reading camera
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert color mg to gray image
    face=lod_alg.detectMultiScale(grayimg,1.3,4) #getting face quardinate
    for (x,y,w,h) in face:
        cv2.rectangle (img,(x,y),(x+w,y+h),(150,0,255),2) #(150,0,255) represent color,2 represent width of border
    cv2.imshow ("FACEDETECTING",img)  
    key=cv2.waitKey(10)
    if key ==27: #press 'esc' key to escape
        break
cam.release()
cv2.destroyAllWindows()


#by -- SIVARAJ C 
#GETHUB -- Sivarajc2005
#instagram -- sivarajc2005
#linked in -- www.linkedin.com/in/sivaraj-c2005