# multimodal-emotion-detection# Realtime Emotion Detection Using Keras

### This model has been trained for 40 epochs and runs at 71.69% accuracy.
Install the librariess below using pip or conda:
* pip install numpy
* pip install pandas
* pip install tensorflow
* pip install keras
* pip install opencv-python
* pip install future (used for Tkinter)

Download HAAR-Cascade file from :
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 

#### Download the Dataset here:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

#### First run main.py to create and train the model:
```
python main.py
```
#### Then run UI.py for implementing the face-emotion recognition interface:
```
python UI.py
```
This project has made use of FER (Facial Emotion Detection) dataset formed and compiled in the year 2013. The original FER2013 dataset in Kaggle is available as a single csv file, here. I had this converted into a dataset of images in the PNG format for training/testing.

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image.
