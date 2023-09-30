import cv2
import pyttsx3

thres = 0.45  # Threshold to detect object

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)  # Change the camera index if needed
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    
    if success:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                detected_class = classNames[classId - 1].upper()
                cv2.putText(img, detected_class, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Generate audio response when an object is detected
                engine.say(f"Detected {detected_class}")
                engine.runAndWait()

        cv2.imshow("Output", img)
    else:
        print("Failed to read from the camera.")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
