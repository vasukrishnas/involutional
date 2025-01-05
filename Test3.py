# ------ importing Packages ------- #
import numpy as np
from tensorflow.keras.models import load_model
from __pycache__.config import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import warnings
import  cv2
warnings.filterwarnings("ignore")

# ------ reading data ------- #
Tk().withdraw()
print()
print("Select the Input image")
print()
image_path = askopenfilename(initialdir='Testing/')

img=cv2.imread(image_path)
img =cv2.resize(img, (224, 224))
cv2.imshow("Original",img)

""" 1. ------------ PreProcessing ------------ """

# ------ combined tri-lateral guided filtering (CTri-LGF) -------- #
import Preprocessing3
trilateralguided = Preprocessing3.CTri_LGF(img)
cv2.imshow("PreProcessing",trilateralguided)

""" 2. ------------ ROI Extraction ------------ """

# ------ Hexagonal shaped ROI extraction ------ #
import ROI3
Extraction = ROI3.roi_extraction(trilateralguided)
cv2.imshow("ROI",Extraction)

"""3. ------------ Feature extraction ----------"""

# ------ laplace Transform -------- #
import Feature_Extraction3
FeaExt_Img = Feature_Extraction3.laplace_transf(Extraction)
cv2.imshow("Laplace",FeaExt_Img)

""" 4. ------------ Clasification ------------"""

classes=["Female"," Male"]
#2. Dilated Axial Attention convolutional neural network (DAA-CNN)
import DAA_CNN3
classification_2 = load_model("model_saved/Ex_CNN")
classification_2= load_model(image_path)
y_predict = classification_2.predict(FeaExt_Img)
predicted = classification_2.predicted(FeaExt_Img)
print("DAA-CNN:")
print(y_predict,"--->",classes[predicted])


# Self-spectral attention-based relational transformer Net (SSA-RTNet)
import SSARTNet_classify
classification_4 = load_model("model_saved/SSARTNet")

classification_4= load_model(image_path)
y_predict = classification_4.predict(FeaExt_Img)
predicted = classification_4.predicted(FeaExt_Img)
print("SSARTNet:")
print(y_predict,"--->",classes[predicted])

# Parameterized Hypercomplex Convolutional Siamese network (PHCSN)
import PHCSN_classify
classification_5 = load_model("model_saved/PHCSN")

classification_5= load_model(image_path)
y_predict = classification_5.predict(FeaExt_Img)
predicted = classification_5.predicted(FeaExt_Img)
print("PHCSN:")
print(y_predict,"--->",classes[predicted])


# ------ OverAll Performance -------- #
import performance3
Result = performance3.plot()








