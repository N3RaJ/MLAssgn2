I have used wrapper,ensemble and embedded methods to try and extract best features 
Gene is not only the most important feature but it also has most null values, which if filled with 1 has a significant boost on the accuracy 
This is termed as MNAR - Missing not at random 
The pcr rate file contains code for visualising - the features with most null values and their impact on pCR 
The Feature Imp graph visualises in rfs.py ties all understanding together
I have tested the model, via setting null values with 1 and simply nan to be later imputed, 
Nan to 1 gives an accuracy of 80
imputing nan gives an accuracy of 74
if this classification model is selected for our project, we can choose the strategy to use depending on the perfomance on test dataset; as it is simply effective and can be done at any point 
