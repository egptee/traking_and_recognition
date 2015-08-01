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
face_cascade = cv2.CascadeClassifier('path to the classifier : haarcascade_frontalface_alt_tree.xml(suggerito)')


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




pathdir='prove/'
#questavolta le facce dello stream

#inizializzazione:
quanti = int(raw_input('Quanti siete davanti alla webcam? \n numero:'))
for i in range(quanti):
    nome = raw_input('Ciao utente '+str(i+1)+' qual è il tuo nome?\n nome:')
    if not os.path.exists(pathdir+nome): os.makedirs(pathdir+nome)
    print ( 'sei pronto per farmi scattare qualche foto? \n')
    print ( ' ci vorranno solo 10 secondi\n premi "S" quando sei al centro ')
    while (1):
        ret,frame = vc.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Recognition',frame)

        
        if cv2.waitKey(10) == ord('s'):
            break
    cv2.destroyAllWindows()

    #comincio a scattare
    start = time.time()
    count = 0
    while int(time.time()-start) <= 14:
        
        ret,frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            cv2.putText(frame,'Click!', (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
            count +=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            resized_image = cv2.resize(frame[y:y+h,x:x+w], (273, 273))
            if count%5 == 0:
                print  pathdir+nome+str(time.time()-start)+'.jpg'
                cv2.imwrite( pathdir+nome+'/'+str(time.time()-start)+'.jpg', resized_image );
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

    for (x,y,w,h) in faces:
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        sampleImage = gray[y:y+h, x:x+w]
        sampleImage = cv2.resize(sampleImage, (256,256))

        #capiamo di chi è sta faccia
        [ predicted_label, generic_classifier_output] = model.predict(sampleImage)
        print [ predicted_label, generic_classifier_output]
        #scelta la soglia a 700. soglia maggiore di 700, accuratezza minore e v.v.
        if int(generic_classifier_output['distances']) <=  700:
            cv2.putText(img,'tu sei : '+str(subject_dictionary[predicted_label]), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
    cv2.imshow('result',img)
    if cv2.waitKey(10) == 27:
        break



cv2.destroyAllWindows()
vc.release()


