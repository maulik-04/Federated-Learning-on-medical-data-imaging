# Federated Learning on Medical Data Imaging

I have worked on Federated Learning on Medical Data Imaging in this project. For this, I have used the BraTS-2020 dataset. Doctors use MRI scans to detect tumour size and shape, but doing it manually takes time and effort. So we can automate it so that the machine can detect the size of the tumour by looking at MRI scans. For this, we can train a machine-learning model. We can pool the hospitals' data and train a centralised model.

However, the problem is that hospitals cannot share their sensitive patient data with the centralised server because of guidelines like HIPAA and GDPR. This can also create the problem of data silos - isolated datasets confined to the institution.

Therefore, we use the concept of Federated Learning. It is a decentralised machine learning paradigm where institutions can train their models locally, and then the model updates can be aggregated and used for the centralised model. So, instead of sending the data to the server, we send the ML algorithm to the local institutions.

**Models Used for Training**:- I have used the UNet and Attention UNet architectures to train the models. Both are designed for biomedical image segmentation tasks.

## BraTS 2020 Dataset

It contains 369 brain MRI scans from glioma patients. Each scan has 4 MRI modalities : 

1. T1-weighted (T1)
2. Contrast-enhanced T1-weighted (T1CE)
3. T2-weighted (T2)
4. Fluid-attenuated inversion recovery (FLAIR)
   
Each scan contains a segmentation mask

0. Background (Purple Region)
1. Necrotic and Non Enhancing Tumour Core (Blue Region)
2. Peritumoral Edema (Green Region)
3. GD Enhancing Tumour (Yellow Region)
   
![Segmentation Mask](https://github.com/user-attachments/assets/5cfceb1d-105c-45ed-a5a7-eb931fff0559)

## Preprocessing Steps

Preprocessing mainly includes 3 things :- Scaling, cropping and combining the channels.

I have used MinMax Scaling to scale the images. 

From the original dataset, the image shape that we get is (240,240,155), so I have cropped the images to (128,128,128). This is because as you can see in the segmentation mask, most of the outer region just has the background, so we can safely crop it so that the model can focus on the tumour region.

Then I merged the 3 channels. I have ignored T1 channel because T1ce is just the contrast enhanced version of T1, so there is no point of doing repetition.

## U Net Architecture

![](https://github.com/user-attachments/assets/83f2e225-e3e2-4b44-ba55-be22b855c8a4)

## Attention Unet Architecture

![](https://github.com/user-attachments/assets/9c7a6266-b385-4842-b060-a236d0158cce)


## Federated Learning



**Federated Learning Workflow** :- 

Step 1 :- The server creates a unet model with random weights. 

Step 2 :- Server sends the global model to all 5 hospitals.

Step 3 :- Each hospital trains the model on their private MRI scans, computes model updates and then sends only the updated weights to the server.

Step 4 :- Server averages the weights from all hospitals.

Step 5 :- Server replaces the old global model with the averaged weights.

We will repeat steps 2-5 for multiple rounds.

## Results 

![](result.png)

Refer to the report for full understanding.
