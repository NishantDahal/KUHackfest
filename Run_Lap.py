import cv2
import pyttsx3
import threading
import time  # Import the time module

# Threshold to detect object
thres = 0.60

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Speed up the audio response (adjust the rate as needed)
engine.setProperty('rate', 250)  # You can experiment with different values

cap = cv2.VideoCapture(0)  # Change the camera index if needed

# Set the desired frame rate (5 fps)
cap.set(cv2.CAP_PROP_FPS, 1)

# Set the desired resolution (640x360)
cap.set(3, 640)
cap.set(4, 360)

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

# Dictionary to store the time of the last announcement for each class
last_announcement_time = {}

# Function to handle text-to-speech in a separate thread
def speak_async(text):
    engine.say(text)
    engine.runAndWait()

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

                # Check if it's been more than 10 seconds since the last announcement for this class
                current_time = time.time()
                last_time = last_announcement_time.get(detected_class, 0)
                if current_time - last_time >= 4:
                    # Generate audio response in a separate thread
                    threading.Thread(target=speak_async, args=(f"{detected_class}",)).start()

                    # Update the last announcement time for this class
                    last_announcement_time[detected_class] = current_time

        cv2.imshow("Output", img)
    else:
        print("Failed to read from the camera.")
    
    if cv2.waitKey(200) & 0xFF == ord('q'):  # Delay to achieve 5 fps
        break

cap.release()
cv2.destroyAllWindows()
