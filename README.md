# Pytorch_Audio-Emotion-Classifier
Training a model using CNN's to predict the emotion class of an Audio file in pytorch framework.

Audio classification can be used to interpret audio scenario, which is critical in turn for an artificial entity
to understand and communicate more efficiently with its environment. This project is a successful
implementation of simple Convolutional Neural Networks in pytorch which can predict the emotion of
random audio file with 60 percent accuracy. There is a scope to improve the performance by building a
robust model and by augmentation of audio files by noise injection, time shift or stretching and pitch
shifting.

# Audio file Tranformations
To make the model understand all the information present in speech signals we have to convert audio in time domain(analog) 
into frequency domain. But just converting time domain signals into frequency domain may not be very optimal. We can do more 
than just converting time domain signals into frequency domain signals. Our ear has cochlea which basically has more filters
at low frequency and very few filters at higher frequency. This can be mimicked using Mel filters. So, the idea of MFCC is to 
convert time domain signals into frequency domain signal by mimicking cochlea function using Mel filters. Padding
is done to set length of all torch tensors to same length.

# Simple Convolutional Network
![Screenshot (7)](https://user-images.githubusercontent.com/70043813/90946705-73f6b300-e3fd-11ea-8c86-d82f1803200b.png)
Could have worked on much sophisticated model but the data size on which we are training a simple network is taking a whole day to run.
Much bigger and better model would fetch best results but that goes out of scope in aspects of time and computational expenses for an academic project.

