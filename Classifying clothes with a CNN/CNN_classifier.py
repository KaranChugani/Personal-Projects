# -*- coding: utf-8 -*-
"""
This function trains the CNN model using the best optimizer function selected in 
HyperparameterTuning.py ("Adam" in this case) and evaluates it on the testing 
dataset
"""
import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from CNN_functions import create_model2

#%% IMPORTING DATA AND RESHAPING IT FOR INPUT IN CNN 

mnist = tf.keras.datasets.fashion_mnist
(TrainData, TrainLabels), (TestData, TestLabels) = mnist.load_data()
# Reshaping data
TrainData=TrainData.reshape(60000, 28, 28, 1)
TestData=TestData.reshape(10000, 28, 28, 1)
# Normalizing pixels
TrainData=TrainData / 255.0
TestData=TestData / 255.0



#%% TRAINING AND TESING MODEL WITH BEST OPTIMIZER FUNCTION

model=create_model2('Adam',80)
model.fit(TrainData, TrainLabels, epochs=5)
Classification = model.predict(TestData)
PredictionLabels = Classification.argmax(axis=-1)
Accuracy = accuracy_score(TestLabels,PredictionLabels)


#%% CONFUSION MATRIX PLOT OF RESULTS


# Calculating prediction labels and creating a confusion matrix
cm = confusion_matrix(TestLabels,PredictionLabels) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot'], 
                     columns = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot'])

# Plotting the figure
plt.figure
sns.heatmap(cm_df, annot=True)
plt.title('CNN \nAccuracy:{0:.3f}'.format(accuracy_score(TestLabels,PredictionLabels)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
b, t = plt.ylim() 
b += 0.5
t -= 0.5
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


