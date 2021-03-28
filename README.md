# Gesture Recognition using Convolutional Neural Networks

American Sign Language (ASL) is a complete, complex language that employs signs made by moving the hands combined with facial expressions and postures of the body. It is the primary language of many North Americans who are deaf and is one of several communication options used by people who are deaf or hard-of-hearing.  

The hand gestures representing English alphabet are shown below. This project focuses on classifying these hand gesture images using convolutional neural networks. Specifically, given an image of a hand showing one of the letters, we want to detect which letter is being represented.  
![Alt text](/../main/images/symbols.png?raw=true)  

Without transfer learning, and using my own model, I was able to obtain about 55% accuracy on the validation set. The loss and accuracy plots are shown below.  
![Alt text](/../main/images/modelloss.png?raw=true) ![Alt text](/../main/images/modelaccuracy.png?raw=true)  

When using the VGG16 pretrained model and adding additional layers to it, I obtained about a 95% accuracy (VGG16 has 14,714,688 trainable parameters, creating my own model of this size would be impractical and very difficult to train). The loss and accuracy plots are shown below.  

![Alt text](/../main/images/vgg16lossacc.png?raw=true)
