from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from pygame import mixer #ini untuk audio nya menggunakan mixer yang diambil dari pygame
import numpy as np
import imutils
import time
import cv2
import os

#untuk menyiapkan music player
mixer.init()

#untuk mendetek dan memprediksi masker
def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), #untuk mengnormalisasi image
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob) #untuk mendeteksi wajah
	detections = faceNet.forward() #untuk menyimpan hasil lokasi dari deteksi wajah di variabel detections
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	#untuk menampilkan frame hijau di wajah kita dan keterangan akurasinya
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
	
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#data prepocessing
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))
	#jika terdeteksi wajah, mengprediksi ada masker atau tidak
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	
	return (locs, preds) #mereturn nilai lokasi wajah dan prediksi wajah tersebut menggunakan masker/tidak

prototxtPath = r"face_detector\\deploy.prototxt"
weightsPath = r"face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath) #model untuk mendeteksi wajah

maskNet = load_model("mask_detector.model")

print("Starting...")
vs = VideoStream(src=0).start() # untuk memulai videostream 

no_mask_iter = 0
mask_iter = 0

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	#menproses semua muka yang terdeteksi
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		#jika akurasi mask lebih tinggi daripada without mask maka akan dikategorikan menggunakan masker dan sebaliknya
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		if label == "No Mask": 
			no_mask_iter += 1
			mask_iter = 0
		elif label == "Mask":
			mask_iter += 1
			no_mask_iter = 0

		#mengatur bunyi dan sensitifitas nya ketika tidak menggunakan masker
		sensitivityNoMask = no_mask_iter > 10 # 10 frame
		sensitivityMask = mask_iter > 10 # 10 frame
		if label == "No Mask" and sensitivityNoMask and not mixer.music.get_busy():
			no_mask_iter = 0
			mask_iter = 0
			mixer.music.load("sirine.mp3")
			mixer.music.stop()
			mixer.music.play()
		#ini untuk yang menggunakan masker
		elif label == "Mask" and sensitivityMask:
			no_mask_iter = 0
			mask_iter = 0
			mixer.music.load("Unlock.wav")
			mixer.music.stop()
			mixer.music.play()

		
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"): #exit program ketika memencet q
		break

cv2.destroyAllWindows()
vs.stop()