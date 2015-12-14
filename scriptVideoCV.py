from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from PIL import Image
import numpy as np
from PIL import Image
import sys, os
import time
#sys.path.append("../..")
import cv2
import multiprocessing



model = PredictableModel(Fisherfaces(), NearestNeighbor())

vc=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/Users/yuxiao/haarcascade_frontalface_alt.xml')
if face_cascade.empty():
    print('can not find haarcascade_frontalface_alt.xml')


#una volta ottenuto (prossimo step) un db di facce, le 
def read_images(path, sz=(256,256)):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    folder_names = []
    
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]



def modifyparam(scale,model):
    ret,frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, model)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Recognition',frame)
        
    cv2.destroyAllWindows()



pathdir='prove/'
#questavolta le facce dello stream

def ismatch(x1,y1,w1,h1,x2,y2,w2,h2,margin=5):
    if abs(x1-x2)<margin and abs(y1-y2)<margin and abs(w1-w2)<margin and abs(h1-h2)<margin:
        return True
    else:
        return False


#inizializzazione:
quanti = int(raw_input('how many people to recognize? \n number:'))
for i in range(quanti):
    nome = raw_input('welcome '+str(i+1)+' what\'s your name?\n nome:')
    if not os.path.exists(pathdir+nome): os.makedirs(pathdir+nome)
    print ( 'be ready for me to take some photos of you \n')
    print ( ' it only takes 10 seconds\n press "S" if you are in the rect')
    
    # while (1):
    #     ret,frame = vc.read()
        # size=frame.shape[:2]
        # h, w = size
        # minSize=(int(w*0.3), int(h*0.5))
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray,1.1,4,0,minSize)
    #     for (x,y,w,h) in faces:
    #         cv2.rectangle(frame,(300,50),(980,670),(0,0,255),2)
    #         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     cv2.imshow('Recognition',frame)
    #     if cv2.waitKey(10) == ord('s'):
    #         break
    # cv2.destroyAllWindows()

    sd_x = 400
    sd_y = 120
    # 480*480
    sd_w = 1280 -2*sd_x 
    sd_h = 720 - 2*sd_y
    margin = 5
    #comincio a scattare
    start = time.time()
    count = 0
    hittime=0
    prehit=hittime
    while (1):
        ret,frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size=frame.shape[:2]
        h, w = size
        minSize=(int(w*0.3), int(h*0.5))
        faces = face_cascade.detectMultiScale(gray,1.1,4,0,minSize)
        for (x,y,w,h) in faces:
            key =cv2.waitKey(10)
            if ismatch(x,y,w,h,sd_x,sd_y,sd_w,sd_h,margin=20):
                count += 1
                if (count + 1) % 5 == 0:
                    prehit = hittime
                    hittime += 1
                    resized_image = cv2.resize(frame[y:y+h,x:x+w], (273, 273))
                    print  pathdir+nome+'/'+str(time.time()-start)+'.jpg'
                    cv2.imwrite( pathdir+nome+'/'+str(time.time()-start)+'.jpg', resized_image );
                    break;
            cv2.putText(frame,'hit '+str(hittime)+' times', (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #break;
        if hittime > 5:
            break;
        if hittime != prehit:
            prehit = hittime
            cv2.rectangle(frame,(sd_x,sd_y),(sd_x+sd_w,sd_y+sd_h),(255,255,255),5)
        else:
            cv2.rectangle(frame,(sd_x,sd_y),(sd_x+sd_w,sd_y+sd_h),(0,0,255),2)
        cv2.imshow('Recognition',frame)
        cv2.waitKey(10)
    cv2.destroyAllWindows()



[X,y,subject_names] = read_images(pathdir)
list_of_labels = list(xrange(max(y)+1))

subject_dictionary = dict(zip(list_of_labels, subject_names))
model.compute(X,y)

#comincia il riconoscimento.
while (1):
    rval, frame = vc.read()
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    cv2.rectangle(frame,(sd_x,sd_y),(sd_x+sd_w,sd_y+sd_h),(0,0,255),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleImage = gray[y:y+h, x:x+w]
        sampleImage = cv2.resize(sampleImage, (256,256))

        #capiamo di chi è sta faccia
        [ predicted_label, generic_classifier_output] = model.predict(sampleImage)
        print [ predicted_label, generic_classifier_output]
        #scelta la soglia a 700. soglia maggiore di 700, accuratezza minore e v.v.
        confi = generic_classifier_output['distances']
        if int(confi) >=  5000:
            cv2.putText(img,'guess: '+str(subject_dictionary[predicted_label])+" "+str(confi), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
        elif int(confi) >=  3000:
            cv2.putText(img,'strong: '+str(subject_dictionary[predicted_label])+" "+str(confi), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
        elif int(confi) <=  700:
            cv2.putText(img,'confirm: '+str(subject_dictionary[predicted_label])+" "+str(confi), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
        else :
            cv2.putText(img,'bit confirm: '+str(subject_dictionary[predicted_label])+" "+str(confi), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
    cv2.imshow('result',img)
    if cv2.waitKey(10) == 27:
        break


cv2.destroyAllWindows()
vc.release()


