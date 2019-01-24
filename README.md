# Whale - Kaggle competition
Try to predict type of whale by reading the image of whale tail
## Challenge
There are 5005 types of whale in the dataset. Number of images of type "new_whale" is more than 9000 which is 40%. And For each type of whale, there are only 3 to 5 images

## Transfer learning
Use pre-trained VGG16 CNN and add 2 fully connected layers.
Run the code on WPI Turing Research Cluster.

## Step
### 1: Run with raw data 
(Current stage) Training accuracy is only 38% and the loss stop decreasing after 9th iteration.
### 2: Augment images
for each type except "new_whale", use Python library imgaug to create more images
### 3: To be continued
