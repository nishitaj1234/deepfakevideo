import cv2
import dlib
camera=cv2.VideoCapture("Taylor Swift/Deepfakes/taylorfake3.mp4")
camera.set(3,600)
camera.set(4,400)

detector=dlib.get_frontal_face_detector()
landmarkpred=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
file= open("Taylor Swift/Deepfakes/taylorfake3.txt","w")
for n in range(0,68,1):
    file.write("x"+str(n)+","+"y"+str(n)+",")
file.write("ratio"+","+"class")
file.write("\n")

while camera.isOpened():
    status,frame=(camera.read())
    print(status)
    if status==True:
        smallframe=cv2.resize(frame,(600,400))
        greyimage=cv2.cvtColor(smallframe,cv2.COLOR_BGR2GRAY)
        faces=detector(greyimage)
        for n in faces: 
            landmarks=landmarkpred(greyimage,n)
            points=[]
            for facialpoint in range(0,68,1):
                xcoor=landmarks.part(facialpoint).x
                ycoor=landmarks.part(facialpoint).y
                file.write(str(xcoor))
                file.write(",")
                file.write(str(ycoor))
                file.write(",") 
                points.append([xcoor,ycoor])
                cv2.circle(smallframe, (xcoor,ycoor), 1, (255,0,0), 2)
                cv2.putText(smallframe,str(facialpoint), (xcoor,ycoor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            xdistance=points[16][0]-points[0][0]
            ydistance=points[8][1]-points[19][1]
            file.write(str(xdistance/ydistance))
            file.write(",")
            file.write("1")
            file.write("\n")

        cv2.imshow("frame",smallframe)
    if cv2.waitKey(30) & 0xFF==ord("q"):
        break
 

camera.release()
cv2.destroyAllWindows()

file.close()



'''for n in faces: 
        landmarks=landmarkpred(greyimage,n)
        points=[]
        for facialpoint in range(0,68,1):
            xcoor=landmarks.part(facialpoint).x
            ycoor=landmarks.part(facialpoint).y
            points.append([xcoor,ycoor])
            cv2.circle(image, (xcoor,ycoor), 1, (255,0,0), 0)
        #points=np.array(points10)
        #taylormouth,taylorboxmouth,taylormouthroi=makebox(image2,points10,str(n),"taylor")

cv2.imshow("image",image)
cv2.waitKey()
cv2.destroyAllWindows()'''
