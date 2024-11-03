from facial_emotion_recognition import EmotionRecognition
import cv2
er=EmotionRecognition(device='cpu')
vs=cv2.VideoCapture(0)
while True:
    a,img=vs.read()
    recog=er.recognise_emotion(img,return_type='BGR')
    cv2.imshow('show',recog)
    key=cv2.waitKey(10)
    if key == ord('x'):
        break
vs.release()
cv2.destroyAllWindows()
