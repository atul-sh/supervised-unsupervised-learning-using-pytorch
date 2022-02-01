# supervised-unsupervised-learning-using-pytorch
# CIFAR-10 dataset Image Classification using AutoEncoder as Feature Extractor

A) The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. * There are 50000 training images and 10000 test images.

B)     The dataset is divided into five training batches and one test batch, each with 10000 images.

C)     The test batch contains exactly 1000 randomly-selected images from each class.

D)     The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

E)     These are the classes in the dataset:

        1. airplane
        2. automobile
        3. bird
        4. cat
        5. deer
        6. dog
        7. frog
        8. horse
        9. ship
        10.truck

F) The classes are completely mutually exclusive. i.e. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.


##### Training output

    
![png](output_36_0.png)
    



    
![png](output_37_0.png)
    






##### Compare the ground truth vs Prediction




    Ground Truth : Predicted 



    
![png](output_45_1.png)
    


##### Check Accuracy of the netwrok



    Accuracy of the network on the 10000 test images: 60 %


##### Cheeck the accuracy of each class in the dataset



    Accuracy for class: plane is 71.0 %  [710/1000]
    Accuracy for class: car   is 81.6 %  [816/1000]
    Accuracy for class: bird  is 27.9 %  [279/1000]
    Accuracy for class: cat   is 41.4 %  [414/1000]
    Accuracy for class: deer  is 31.4 %  [314/1000]
    Accuracy for class: dog   is 75.5 %  [755/1000]
    Accuracy for class: frog  is 75.9 %  [759/1000]
    Accuracy for class: horse is 75.5 %  [755/1000]
    Accuracy for class: ship  is 64.4 %  [644/1000]
    Accuracy for class: truck is 62.7 %  [627/1000]


##### plot the Confusion matrix of the predicted score




    
![png](output_51_0.png)
    


##### classifiacation report of the data



                  precision    recall  f1-score   support
    
           plane       0.66      0.72      0.69      1000
             car       0.75      0.81      0.78      1000
            bird       0.62      0.28      0.38      1000
             cat       0.38      0.42      0.40      1000
            deer       0.71      0.31      0.43      1000
             dog       0.38      0.74      0.50      1000
            frog       0.66      0.75      0.70      1000
           horse       0.60      0.75      0.66      1000
            ship       0.86      0.65      0.74      1000
           truck       0.83      0.61      0.71      1000
    
        accuracy                           0.60     10000
       macro avg       0.64      0.60      0.60     10000
    weighted avg       0.64      0.60      0.60     10000
    





    
![png](output_56_1.png)
    

### end
