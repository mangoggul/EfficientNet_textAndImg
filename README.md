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

```bash
#data_dir/
#│
#├── train/
#│   ├── rgb/안에 ca1 ca2 폴더
#│   │   ├── 이미지 파일들 (예: img1.jpg, img2.jpg, ...)
#│   └── labels/안에 ca1 ca2 폴더
#│       ├── 라벨 파일들 (예: img1.json, img2.json, ...)
#│
#├── val/
#│   ├── rgb/안에 ca1 ca2 폴더
#│   │   ├── 이미지 파일들 (예: img1.jpg, img2.jpg, ...)
#│   └── labels/안에 ca1 ca2 폴더
#│       ├── 라벨 파일들 (예: img1.json, img2.json, ...)
#│
#└── test/
#    ├── rgb/안에 ca1 ca2 폴더
#    │   ├── 이미지 파일들 (예: img1.jpg, img2.jpg, ...)
#    └── labels/안에 ca1 ca2 폴더
#        ├── 라벨 파일들 (예: img1.json, img2.json, ...)
 
``` 


> It doesn't matter whether the images are in JPG or PNG format.

### 3. Start Training
type this python command
<br/>
if you want single modal rgb image
```
python single_modal_train.py
```
if you want multi modal image
```
python multi_modal_train.py
```



### 4. Inference 
After Training you can use inference.ipynb file. 

![image](https://github.com/user-attachments/assets/da8e245f-86ff-4130-9183-f14ba6cf3c5f)

Furthermore, metrics

![image](https://github.com/user-attachments/assets/f00f8fec-b42f-45c1-a16f-9965652c0d12)

### 5. Compare with Single Modal & Multi Modal


|  | SingleModal (Rgb)| MultiModal (Rgb And Depth) |
| --- | --- | --- |
| Accuracy | 0.94444 | 0.96296 |
| Precision | 0.95370 | 0.96732 |
| Recall | 0.94444 | 0.96296 |
| F1 Score | 0.94445 | 0.96309 |

**100 epoch Trained**


