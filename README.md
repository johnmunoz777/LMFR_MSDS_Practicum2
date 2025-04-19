# Live Member Face Recognition Deployment
## Project Description
The purpose of this project is to develop a more efficient way to verify memberships and grant access to members without unnecessary delays. <br>
This issue exists in places like Costco, sporting events, and other venues where long queues form just to validate entry credentials.<br>
The goal is to eliminate long wait times by allowing members to gain access seamlessly through facial recognition, reducing the need for manual verification
<!-- 3‑column CSS grid. Fill in as many <figure> blocks as you have images. -->
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; justify-items: center;">
  <figure style="margin: 0; text-align: center;">
    <img src="images/costco_use.png" alt="Members Example 1" width="250" height="200" />
    <figcaption style="font-size: 0.75em; color: #555;"></figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="images/costco_use_two.jpg" alt="Members Example 2" width="250" height="200" />
    <figcaption style="font-size: 0.75em; color: #555;"></figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="images/costco_use_three.jpg" alt="Members Example 3" width="250" height="200" />
    <figcaption style="font-size: 0.75em; color: #555;"></figcaption>
  </figure>
  <!-- Duplicate or add more <figure> blocks here for a full 3×3 grid -->
</div>
<br>
Found on Pexels.com  

## Project Proposal

To solve this problem, I developed a real-time face recognition system using computer vision.  
In [Practicum 1: YOLOv8 Model](https://github.com/johnmunoz777/LMFD_MSDS_Practicum) I built a YOLOv8 face recognition model.  
The purpose of Practicum 2 was to deploy this model so anyone can add themselves to it  <br>
and employees can validate members in real time via a webcam on our website:[Practicum 2: Deployment Demo](https://facedetectiondemo.com/)

<img src="images/results_hugging.gif" alt="john Example" width="800" height="400" />
<br>

<!-- Image 2 -->
<img src="images/phone.jpg" alt="Project" width="600" height="500" />
<br>



<img src="images/phoneview.jpg" alt="Project" width="600" height="500" />

## System Overview
This system leverages an sqlite members database, OpenCV, YOLO for object detection, Hugging Face Spaces for the dahsboard, Cloudflared,and Flask for the employee website <br>
By implementing this solution, venues such as Costco and retail stores can have frictionless, secure, and efficient entry for their members.

<img src="images/flowchart_hugging_face.png" alt="john Example" width="700" height="400" />
<br>
<img src="images/tunnel_process.jpg" alt="Project" width="700" height="600" />
<br>

### Table of Contents  
- [Yolo Model Comparison](#Model-Comparison)  
- [Setting Up Hugging Face Space](#Setting-Up-Hugging-Face-Space)  
- [Hugging Face Space](#Hugging-Face-Space)  
- [Face Detect Website](#Face-Detect-Website)
- [File Information](#File-Information)
- [Future Implementations](#future-implementations)


### Yolo Model Comparison

### Models Trained
I developed two final models for my project. <br>
* Large Yolo Model- this Model contained 14 individuals and had on average around 300 images per class
* Small Yolo Model- this Model contained 7 individuals which had had on average around 100 images per class

# Prepare Yolo Data Large Model Example

To prepare the images for the YOLO v8 object detection model, I used OpenCV to detect faces within the images. Specifically, I employed the Haar Cascade classifier with the haarcascade_frontalface , an algorithm widely used for object detection, especially for faces. Once a face was detected, I generated a `.txt` file containing the volunteer's ID, the `x_center`, `y_center`, `width`, and `height` of the detected face.

I created a Python function that:

- Splits the images into training, validation, and test sets according to specified split ratios.
- Generates YOLO-formatted `.txt` annotation files for each image by detecting faces.
- Copies both the images and annotation files into their corresponding output folders.
- Finaly, data.yaml file linked the path,train,validation, number_of_classes

The folder structure for the output is as follows:
```bash
output_base/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

        
```
# Dataset Paths and Configuration
* This is an Example of how I setup the Large Yolo Model
* The Large Yolo Model had 14 individuals in contrast the Small Yolo Model contained 7 Individuals

```yaml
path: /content/drive/MyDrive/capstone project/my_data_split
train: /content/drive/MyDrive/capstone project/my_data_split/train/images
val: /content/drive/MyDrive/capstone project/my_data_split/val/images
test: /content/drive/MyDrive/capstone project/my_data_split/test/images
nc: 14
names: ['angela', 'classmate', 'giuliana', 'javier', 'john', 'maite', 'mike', 'ron', 'shanti', 'tom', 'vilma', 'will','Kevin','Shirley']
```

<img src="images/g_doc.jpg" alt="john Example" width="800" height="400" />

### Building the YOLO Model
I used the Ultralytics package to build the Yolo v8 model <br>
`!yolo task=detect mode=train model=yolov8l.pt data="/content/drive/MyDrive/capstone project/data.yaml" epochs=50 imgsz=640`
![rs](images/yolo_results.jpg)
- Overall Model Performance

Large Model mAP@50: 90% → On average, the Yolo v8 model detects objects correctly 74.5% of the time at an IoU threshold of 0.50. <br>
Small Model mAP@50: 85% → On average, the Yolo v8 model detects objects correctly 74.5% of the time at an IoU threshold of 0.50.<br>
Large Model Precision (P): 89% → Out of all detected objects, 74% were correctly classified.  <br>
Large Model Precision (P): 81% → Out of all detected objects, 74% were correctly classified.   <br>

![rs](images/best_model.jpg)

![rs](images/small_model_results.jpg)

### Confuision Matrix
- Both models achieved very similar **overall accuracy** when evaluated with their confusion matrices.  
- Performance varied by class—some identities had higher **recall** than others.  
- **Large model:**  
  - Angela class **recall** was nearly perfect.  
  - Will class **recall** was the lowest, at 68%.  
- **Small model:**  
  - Dad class **recall** hit 100%, though the limited sample size suggests possible overfitting.  
  - Will class **recall** was the lowest, at 63%.

<img src="images/large_model_confusion_matrix.jpg" alt="Project" width="700" height="400" />
<br>

<img src="images/small_model_confuison_model.jpg" alt="Project" width="700" height="400" />
<br>

## Model Used For Hugging Space
Due to Hugging Face Spaces storage limits, I ultimately used the **small YOLO model**.<br>
The following shows training images for both models 

* Lare Yolo Model
  
<img src="images/train_one.jpg" alt="Project" width="700" height="400" />
<br>

<img src="images/train_two.jpg" alt="Project" width="700" height="400" />
<br>

* Small Yolo Model 

<img src="images/small_model_train.jpg" alt="Project" width="700" height="400" />
<br>

### Setting Up Hugging Face Space
