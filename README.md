# Find-phone
A phone localization project based on CNN
refers to the project provided by YazhouZhang0709(https://github.com/YazhouZhang0709/Object-detection-and-localization-based-on-CNN)

## ***Folder Intro***
### —image
contains the 129 original images for training and testing, and the label——coordinate of the phone, looks like(x, y).

### —data_augmentation
contains the augment images which are flipped from original images, as well as the labels along with flipping. Three kinds of way were used to augment data: horizontally flipping, vertically flipping and diagnally flipping.

### —keras_sln / tensorflow_sln
The solution provided by using keras/tensorflow frame


## ***Usage***
For training: *python train.py*; 
For test: *python test.py*

## ***PS***
1. In keras the accuracy reached 99%(train-acc)/91%(test-acc), while in tensorflow accuracy reached at most 91%(train-acc)/90%(test-acc);
2. file *tf_net_eager* in tensorflow folder is not completed yet for my shortness of tensorflow-eager-execution;
3. Tried to use *autokeras* API to auto_machine_learning a model, however a bug I cannot solve raised which I will try again when official doc update the information about regression.
