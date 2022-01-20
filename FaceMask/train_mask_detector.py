from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#untuk menginisialisasi learnig rate, jumlah epochnya, sama size dari batch  
INIT_LR = 1e-4 
EPOCHS = 20 
BS = 32

#Lokasi File code yang ada di komputer
DIRECTORY = r"E:\\SYSTEM\\GaLaaX\\Codingan\\FaceMask\\dataset"
CATEGORIES = ["with_mask", "without_mask"] #folder

print("Loading Images !!!")

data = [] #untuk mengappend seluruh image data ke list ini
labels = [] #untuk mengappend seluruh gambar yang sesuai dengan menggunakan masker / tidak

#Untuk menglooping data image pada folder categories lalu diubah ukurannya dan ditambahkan datanya ke list data dan labels
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

#untuk mengubah data yang sudah didapat diatas menjadi binary 0 dan 1(0 = with mask, 1 = without mask)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#untuk mengubah data list menjadi array
data = np.array(data, dtype="float32") #diubah menjadi array supaya computer lebih mudah 
										#mempelajarinya karena komputer hanya mengerti angka
labels = np.array(labels)

#untuk mengsplit training dan testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#untuk mengconstruct training image generator untuk data augmentation
aug = ImageDataGenerator( #untuk mengenerate data biar si model ga over fitting
	rotation_range=20,	  #ini process data prepocessing
	zoom_range=0.15, #untuk mengzoom
	width_shift_range=0.2, #untuk menggeser gambar ke kiri atau kekanan
	height_shift_range=0.2, #untuk menggeser gambar dari atas ke bawah
	shear_range=0.15, #sudut nya dimiringkan
	horizontal_flip=True) # untuk di balikkan secara horizontal

#berguna untuk melakukan pelatihan pada data
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel) #untuk mengambil data nilai rata-rata
headModel = Flatten(name="flatten")(headModel) #untuk data nya sebelumnya di ubah menjadi single vector
headModel = Dense(128, activation="relu")(headModel) #untuk menambahkan layer yang fully connected
headModel = Dropout(0.5)(headModel) #untuk mengurangi jumlah neuron
headModel = Dense(2, activation="softmax")(headModel) #berguna untuk memprediksi probabilitas 

#ini yang akan menjadi model yang akan di gunakan untuk training data set
model = Model(inputs=baseModel.input, outputs=headModel)

#untuk mengloop seluruh layer yang ada di base model dan dibekukan/diberhentikan agar 
# tidak terupdate saat first training proses
for layer in baseModel.layers:
	layer.trainable = False

#untuk mengcompile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, #algoritma binary_crossentropy dipake jika label nya hanya 2
	metrics=["accuracy"])

#untuk mengtrain head network
print("[INFO] training head...")
H = model.fit(  #untuk memulai training nya dari dataset
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS, #berguna untuk menentukan berapa banyaknya sample yang digunakan setiap epoch
	validation_data=(testX, testY), #berguna untuk memvalidasi data
	validation_steps=len(testX) // BS, #untuk menentukan berapa banyak nya step ketika berhenti saat validasi
	epochs=EPOCHS)

#untuk membuat sebuah prediksi di testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS) #predIdxs untuk memprediksi sebuah gambar dikategorikan menggunakan masker/tidak

#digunakan untuk menemukan indeks dari label dengan probabilitasa prediksi terbesar
predIdxs = np.argmax(predIdxs, axis=1) #untuk mengambil prediksi dengan probabilitas terbesar

#untuk mengprint classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

#untuk mengserialisasikan model ke disk
print("[INFO] saving mask detector model...") #dikarenakan model nya masih dalam bentuk code sehingga untuk dideploy, 
														#model tersebut di export
model.save("mask_detector.model", save_format="h5")#train data nya akan disave ke file mask_detector.model

#plot the training loss dan accuracy, untuk menampilkan hasil 
N = EPOCHS
plt.style.use("ggplot") 
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")