# ------ importing Packages ------- #
import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    
    # ------ reading data ------- #
    im_path = "UBIPr Database"
    dataset = os.listdir(im_path)
    
    data = []
    Labels = []
    for i in dataset:
        list_1 = im_path + "/" + i
        a = os.listdir(list_1)
        for file in a:
            path = im_path + "/" + i + "/" + file
            
            img=cv2.imread(path)
            img =cv2.resize(img, (256, 256))
            img_normalize = cv2.normalize(img, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
            
            """ 1. ------------ PreProcessing ------------ """
            
            # ------ combined tri-lateral guided filtering (CTri-LGF) -------- #   
            import Preprocessing2
            trilateralguided = Preprocessing2.CTri_LGF(img_normalize)
            Labels.append(i)
            
            """ 2. ------------ ROI Extraction ------------ """
    
            # ------ Hexagonal shaped ROI extraction ------ #
            import ROI2
            Extraction = ROI2.roi_extraction(trilateralguided)
    
            """3. ------------ Feature extraction ----------"""
    
            # ------ Laplace Transform -------- #
            import Feature_Extraction2
            FeaExt_Img = Feature_Extraction2.laplace_transf(Extraction)
            data.append(FeaExt_Img)
          
    data = np.asarray(data)
    Labels = np.asarray(Labels)
        
    """np.save("fs_data2", data)  
    np.save("fs_Label2", Labels) """

# main()

X= np.load("fs_data2.npy")
Y= np.load("fs_Label2.npy")

X_train, X_test,y_train, y_test = train_test_split(X,Y,random_state=104, test_size=0.25, shuffle=True)


X = X_train
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(y_train)
# y_te = lb.fit_transform(y_test)
# y1 = np.argmax(y_te,axis=1)
x = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)


# Self-spectral attention-based relational transformer Net (SSA-RTNet),
# Parameterized Hypercomplex Convolutional Siamese network (PHCSN)

""" 4. ------------ Clasification ------------"""

# Dialated Axial Attention convolutional neural network (Ex-CNN)
import DAA_CNN2
print("---------Classification_2 [DAA-CNN]---------")
classification_2 = DAA_CNN2.DAA_CNN_classifi(x,Y)

# Self-spectral attention-based relational transformer Net (SSA-RTNet)
import SSARTNet_classify
print("---------Classification_4 [SSARTNet]---------")
classification_1 = SSARTNet_classify.SSARTNet_Model(x,Y)

# Parameterized Hypercomplex Convolutional Siamese network (PHCSN)
import PHCSN_classify
print("---------Classification_5 [PHCSN]---------")
classification_5 = PHCSN_classify.PHCSN_model(x,Y)

# -------- Existing {SNN,SAE, Bilstm,CNN,DBN} ---------- #
import Existing2
Ext_model = Existing2.existing_classifiers(x,Y)





