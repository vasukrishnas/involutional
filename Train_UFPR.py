# ------ importing Packages ------- #
import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    
    # ------ reading data ------- #
    im_path = "UFPR-Periocular dataset"
    dataset = os.listdir(im_path)
    
    data = []
    Labels = []
    for i in dataset:
        list_1 = im_path + "/" + i
        a = os.listdir(list_1)
        for file in a:
            path = im_path + "/" + i + "/" + file
            print(file)
            img=cv2.imread(path)
            img =cv2.resize(img, (224, 224))
            img_normalize = cv2.normalize(img, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
            
            """ 1. ------------ PreProcessing ------------ """
            
            # ------ combined tri-lateral guided filtering (CTri-LGF) -------- #   
            import Preprocessing3
            trilateralguided = Preprocessing3.CTri_LGF(img_normalize)
            # Labels.append(i)
            
            """ 2. ------------ ROI Extraction ------------ """
    
            # ------ Hexagonal shaped ROI extraction ------ #
            import ROI3
            Extraction = ROI3.roi_extraction(trilateralguided)
    
            """3. ------------ Feature extraction ----------"""
    
            # ------ laplace Transform -------- #
            import Feature_Extraction3
            FeaExt_Img = Feature_Extraction3.laplace_transf(Extraction)
            data.append(FeaExt_Img)
            Labels.append(i)
          
    data = np.asarray(data)
    Labels = np.asarray(Labels)
        
    """np.save("fs_data", data)  
    np.save("fs_Label", Labels) """
    
# main()

X= np.load("fs_data.npy")
Y= np.load("fs_Label.npy")

X_train, X_test,y_train, y_test = train_test_split(X,Y,random_state=104, test_size=0.25, shuffle=True)

X = X_train
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(y_train)
# y_te = lb.fit_transform(y_test)
x = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)

""" 4. ------------ Clasification ------------"""

#2. Dilated Axial Attention convolutional neural network (DAA-CNN)
import DAA_CNN3
print("---------Classification_2 [Ex-CNN]---------")
classification_2 = DAA_CNN3.Ex_CNN_classifi(X,Y)


# Self-spectral attention-based relational transformer Net (SSA-RTNet)
import SSARTNet_classify
print("---------Classification_4 [SSARTNet]---------")
classification_1 = SSARTNet_classify.SSARTNet_Model(x,Y)

# Parameterized Hypercomplex Convolutional Siamese network (PHCSN)
import PHCSN_classify
print("---------Classification_5 [PHCSN]---------")
classification_5 = PHCSN_classify.PHCSN_model(x,Y)

# -------- Existing {SNN,SAE, Bilstm,CNN,DBN} ---------- #
import Existing3
Ext_model = Existing3.existing_classifiers(x,Y)




