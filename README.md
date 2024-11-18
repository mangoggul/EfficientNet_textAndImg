# EfficientNet_textAndImg

## Flow
![image](https://blog.roboflow.com/content/images/2024/04/image-1203.webp)

I used Multi-Modal method, especially early fusion. 
Here is some kind of 
multi modal fusion. 
## Early Fusion
In this approach, two different types of data are combined into a single dataset before training the model. Various data transformations are performed to merge the two datasets. The data can be concatenated either as raw data or after preprocessing.

## Late Fusion
In this approach, two different types of data are trained separately using different models. The results from these models are then fused together. This method works similarly to boosting in ensemble models.

## Joint (Intermediate) Fusion
This method offers flexibility to merge modalities at a desired depth of the model. It involves progressing with a single modality and then fusing it with another modality just before the final layer of model training. This process is also known as end-to-end learning. 


## How to Use
### 1. git clone this file 

    
```
git clone https://github.com/mangoggul/EfficientNet_textAndImg.git
```

### 2. You need to download Dataset / Dataset is organized as follows

#### Dataset Tree

if I want to classify Ca1 and Ca2

```bash
#data_dir/
#│
#├── train/
#│   ├── rgb/ ca1 ca2 folder
#│   │   ├── Image Files (EX: img1.jpg, img2.jpg, ...)
#│   └── labels/ ca1 ca2 folder
#│       ├── Labels (EX: img1.json, img2.json, ...)
#│
#├── val/
#│   ├── rgb/ ca1 ca2 folder
#│   │   ├── Image Files (EX: img1.jpg, img2.jpg, ...)
#│   └── labels/ ca1 ca2 folder
#│       ├── Labels (EX: img1.json, img2.json, ...)
#│
#└── test/
#    ├── rgb/ ca1 ca2 folder
#    │   ├── Image Files (EX: img1.jpg, img2.jpg, ...)
#    └── labels/ ca1 ca2 folder
#        ├── Labels (EX: img1.json, img2.json, ...)
 
``` 


> It doesn't matter whether the images are in JPG or PNG format.

### 3. Start Training
type this python command
<br/>
if you want single modal rgb image Training
```
python training/single_modal_train.py
```
if you want multi modal image and text Training
```
python training/multi_modal_textRGB.py
```



### 4. Inference 
After Training you can use inference.ipynb file. 

![image](https://github.com/user-attachments/assets/60c62ad5-5dba-444d-afb0-575dfae45a29)

Furthermore, metrics

![image](https://github.com/user-attachments/assets/367bba2e-3b98-4289-aac0-e1ca301f7ae1)

