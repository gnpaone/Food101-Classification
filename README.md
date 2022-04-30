# Food101-Classification

## Table of Contents
+ [About](#about)
+ [Exploratory Data Analysise](#exploratory-data-analysis)
+ [Model](#model)
+ [Training](#training)
+ [Results Evaluation](#results-evaluation)
+ [Room for improvement](#room-for-improvement)
+ [Reference](#reference)

## About

Food101 dataset is a labelled data set with 101 different food classes. Each class contains 1000 images. Using the data provided, a deep learning model built on Keras is trained to classify into various classes in dataset.

__**Note:** The notebook code is a general code and can be used for any dataset__.
<br>**Epoches:** 75
<br>**Batch_size:** 32

Images are split to train and test set with 1000 images per class respectively. 

## Exploratory Data Analysis

Let's preview some of the images.

<img src = "https://github.com/gnpaone/Food101-Classification/blob/main/Images/EDA.png">

The size of the images are mostly consistent, so all the images are scaled to same size, so we dont have to worry about inconsistency.

## Data Augmentation

Since the data set for each class is relatively small to train a good neural network, an image data generator from Keras is used for image tranformation to expand the dataset and to reduce the overfitting problem.

```python
train_datagen = ImageDataGenerator(featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=5,
                 width_shift_range=0.05,
                 height_shift_range=0.05,
                 shear_range=0.2,
                 zoom_range=0.2,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=True,
                 vertical_flip=False,
                 rescale=1/255) #rescale to [0-1], add zoom range of 0.2x and horizontal flip
train_generator = train_datagen.flow_from_directory(
        "content/drive/MyDrive/food-101/food-101/images",
        target_size=(256,256),
        batch_size=32)
test_datagen = ImageDataGenerator(rescale=1/255) # just rescale to [0-1] for testing set
test_generator = test_datagen.flow_from_directory(
        "content/drive/MyDrive/food-101/food-101/images",
        target_size=(256,256),
        batch_size=32)
```
Check the images from data generator. As shown, the images are slightly distorted and rotated. This shall enable the model to learn the important features of the images and produce a more robust model.

<img src = "https://github.com/gnpaone/Food101-Classification/blob/main/Images/datagen.png" width="1000">

## Model
To create a convolution neural network to classfied the images, Keras Sequencial model is used.

```python
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu', input_shape = (256,256,3), kernel_initializer='he_normal'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation = "relu",kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(len(food), activation = "softmax",kernel_initializer='he_normal',kernel_regularizer=l2()))

#callbacks
checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='auto')

model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    epochs = 75,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data = test_generator,
    callbacks=[checkpointer, reduceLR, earlystopping],
    steps_per_epoch=len(dataset)//16,
    validation_steps=len(dataset)//16)
```


**Batch normalisation:** Tested with batch normalisation layers and removed all dropout layers. It results in faster training and higher learning rates, but it caused more overfitting (large diffence between train and test accuracy) than dropout, thus batch normalisation has not been used in this case.

**Optimizers:** *Adam* final accuracy slightly out-performs *RMSProp* and also converge to minima faster as it's similar to *RMSProp + Momentum*.

**Activation Function:** 
*ReLu* activation used at convolution layers to produce a sparse matrix, which requires less computational power then sigmoid or tanh which produce dense matrix. Also, it reduced the likelihood of vanishing gradients. When a>0, the gradient has a constant value, so it results in faster learning than sigmoids as gradients becomes increasingly small as the absolute value of x increases. 
*Softmax* activation used at the last layer to assign the probability of each class.

**Initializers:** Kernal weights are initialized using *He normal* initializers which helps to attain a global minimum of the cost function faster and more efficiently.The weights differ in range depending on the size of the previous layer of neurons and this is a good inializer to be used with *ReLu* activation function.

**Regularization:** *L2 regularization* is implemented aim to decrease the complexity of the model and minimise overfitting by penalising weights with large magnitudes. 


## Training

<img src = "https://github.com/gnpaone/Food101-Classification/blob/main/Images/history.png" width="1000">


Model accuracy managed to increase over each epoch.

## Results Evaluation

Preview some predictions from the model:

<img src = "https://github.com/gnpaone/Food101-Classification/blob/main/Images/prediction.png" width="1000">

The confusion matrix of test images:

<img src = "https://github.com/gnpaone/Food101-Classification/blob/main/Images/cm.png" width="500">

The confusion matrix are of the wrong predictions (confusion matrix looks better if model is trained better, more about it [here](#room-for-improvement)). For visualising the model performance for each class, Receiver Operating Characteristics (ROC) curve is plotted on the true positive rate against false positive rate.

```python
fpr = dict() # false positive rate
tpr = dict() # true positive rate
roc_auc = dict() # area under roc curve
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_conf[:, i]) # roc_curve function apply to binary class only
    roc_auc[i] = auc(fpr[i], tpr[i])  # using the trapezoidal rule to get area under curve

def plot_roc(fpr,tpr,roc_auc):
    plt.figure(figsize=(15,10))
    for i in range(len(food)):
        plt.plot(fpr[i], tpr[i], color='C'+str(i), lw=3, label='ROC curve of {food[i]} (AUC = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--',alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('Receiver Operating Characteristics Curve',fontsize=30)
    plt.legend(loc="lower right",fontsize=15)
    plt.show()

plot_roc(fpr,tpr,roc_auc)
```

Now, let's examine in more detail how the model performs and evaluate those 'wrong-est' predictions.
To determine 'how wrong' the model predicts each images, the wrongly predicted images are sorted by the `difference between the *probability of predicted label* and the *probability of the true class label*`

<img src = "https://github.com/gnpaone/Food101-Classification/blob/main/Images/wrongpredictions.png" width="1000">

## Room for improvement

Due to 1000s of overall images plus lack of time lead to very less accurate model. Accuracy can be improved by increasing number of epochs (thus consuming more time) & training more samples and/or decreasing number of total classes like to 3 or 4 rather than 101 classes.

## Reference

Food 101 Dataset
- [Food-101](https://www.kaggle.com/datasets/jamaliasultanajisha/food101)

<p align="center">
<b>⭐ Please consider starring this repository if it helped you! ⭐</b>
</p>
