from flask import Flask, render_template, request,redirect
import os
import cv2
import dlib
import numpy as np
from keras.models import load_model
model=load_model('deepfaketaylor.h5')
detector=dlib.get_frontal_face_detector()
landmarkpred=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
from werkzeug.utils import secure_filename





app=Flask(__name__)
app.config["UPLOAD_FOLDER"]="./static/uploads"

@app.route("/")
def index():
    return redirect("/upload")

    
@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method=="GET":
        return render_template("upload.html")
    else:
        counter=0
        total=0
        ones=0
        zeroes=0
        file=request.files["file"]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        camera=cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        while camera.isOpened():
            status,frame=(camera.read())
            alldatalist=[]
            if status==True:
                total=total+1
                smallframe=cv2.resize(frame,(600,400))
                greyimage=cv2.cvtColor(smallframe,cv2.COLOR_BGR2GRAY)
                faces=detector(greyimage)
                for n in faces:
                    counter=counter+1
                    landmarks=landmarkpred(greyimage,n)
                    points=[]
                    for facialpoint in range(0,68,1):
                        xcoor=landmarks.part(facialpoint).x
                        ycoor=landmarks.part(facialpoint).y
                        points.append([xcoor,ycoor])
                        alldatalist.append(xcoor)
                        alldatalist.append(ycoor)
                    xdistance=points[16][0]-points[0][0]
                    ydistance=points[8][1]-points[19][1]
                    ratio=xdistance/ydistance
                    alldatalist.append(ratio)
                    alldatalist=np.array(alldatalist)
                    alldatalist=np.reshape(alldatalist,(1,137))
                    prediction=model.predict(alldatalist)
                    if np.argmax(prediction)==0:
                        zeroes=zeroes+1
                    elif np.argmax(prediction)==1:
                        ones=ones+1
            else:
                break
            if cv2.waitKey(30) & 0xFF==ord("q"):
                break
        camera.release()
        cv2.destroyAllWindows()
        if counter==0:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            return render_template("deepfake.html")
        elif counter/total<=0.1:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            return render_template("deepfake.html")
        else:
            if ones>zeroes:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
                return render_template("deepfake.html")
            else:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
                return render_template("notdeepfake.html")
        
if __name__ == "__main__":
    app.run()