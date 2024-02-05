# Neural Network to classify hand-written classes

In this project we created a neural network to classify 8 different classes of hand-written images.
The images are size 28x28 and the following classes can be drawn: \
0: House \
1: Car \
2: In-Ear Headphones \
3: Bottle \
4: On-Ear Headphones \
5: Stick man \
6: TV-Screen \
7: Sun 

## Disclaimer 
This project is a University Project for the Course [Fundamentals of Machine Learning](https://en.unisi.it/ugov/degreecourse/480728) at the [Università degli Studi di Siena](https://www.unisi.it).

## How to use
### Install Requirements
To install all the libraries that we are using run pip install with the requirements.txt

```
pip install -r requirements.txt
```
### Run
To draw on a canvas and draw new images run:
```
python ui.py
```
You draw on the canvas by holding the left mouse button and moving it and can remove white pixels by holding down the right mouse button and moving over the pixels to be removed.
You can also remove all pixels by pressing 'd'.
After drawing an image, you can save the image by pressing a number from 0-7 depending on what label you want to give the image.
This image will then be saved as a training example in the data set.


To display the current dataset and delete examples:
```
python display.py
```
You can go the next image by pressing the right arrow key and go to the previous image by pressing the left arrow key.
To delete the image you're currently seeing, press 'd'. It will be removed from the dataset after you exit the program.

To display the network visualization:
```
python ui_network.py
```
You can move around, zoom in and out and go the next image.



## Authors

* **Silas Ueberschaer** - [Deischox](https://github.com/Deischox)

* **Benjamin Pöhlmann** - [Bepo1337](https://github.com/Bepo1337)
