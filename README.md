![Profile views](https://gpvc.arturio.dev/[aliprf])


# Ad-Corre
Ad-Corre: Adaptive Correlation-Based Loss for Facial Expression Recognition in the Wild

	
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ad-corre-adaptive-correlation-based-loss-for/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=ad-corre-adaptive-correlation-based-loss-for)
<!-- 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ad-corre-adaptive-correlation-based-loss-for/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=ad-corre-adaptive-correlation-based-loss-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ad-corre-adaptive-correlation-based-loss-for/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=ad-corre-adaptive-correlation-based-loss-for)
 -->

#### Link to the paper (open access):
https://ieeexplore.ieee.org/document/9727163

#### Link to the paperswithcode.com:
https://paperswithcode.com/paper/ad-corre-adaptive-correlation-based-loss-for

```
Please cite this work as:

@ARTICLE{9727163,
  author={Fard, Ali Pourramezan and Mahoor, Mohammad H.},
  journal={IEEE Access}, 
  title={Ad-Corre: Adaptive Correlation-Based Loss for Facial Expression Recognition in the Wild}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2022.3156598}}
  
```

## Introduction

Automated Facial Expression Recognition (FER) in the wild using deep neural networks is still challenging due to intra-class variations and inter-class similarities in facial images. Deep Metric Learning (DML) is among the widely used methods to deal with these issues by improving the discriminative power of the learned embedded features. This paper proposes an Adaptive Correlation (Ad-Corre) Loss to guide the network towards generating embedded feature vectors with high correlation for within-class samples and less correlation for between-class samples. Ad-Corre consists of 3 components called Feature Discriminator, Mean Discriminator, and Embedding Discriminator. We design the Feature Discriminator component to guide the network to create the embedded feature vectors to be highly correlated if they belong to a similar class, and less correlated if they belong to different classes. In addition, the Mean Discriminator component leads the network to make the mean embedded feature vectors of different classes to be less similar to each other.We use Xception network as the backbone of our model, and contrary to previous work, we propose an embedding feature space that contains k feature vectors. Then, the Embedding Discriminator component penalizes the network to generate the embedded feature vectors, which are dissimilar.We trained our model using the combination of our proposed loss functions called Ad-Corre Loss jointly with the cross-entropy loss. We achieved a very promising recognition accuracy on AffectNet, RAF-DB, and FER-2013. Our extensive experiments and ablation study indicate the power of our method to cope well with challenging FER tasks in the wild.


## Evaluation and Samples
The following samples are taken from the paper:

![Samples](https://github.com/aliprf/Ad-Corre/blob/main/paper_graphical_items/samples.jpg?raw=true)


----------------------------------------------------------------------------------------------------------------------------------
## Installing the requirements
In order to run the code you need to install python >= 3.5. 
The requirements and the libraries needed to run the code can be installed using the following command:

```
  pip install -r requirements.txt
```


## Using the pre-trained models
The pretrained models for Affectnet, RafDB, and Fer2013 are provided in the [Trained_Models](https://github.com/aliprf/Ad-Corre/tree/main/Trained_Models) folder. You can use the following code to predict the facial emotionn of a facial image:
  
```
    tester = TestModels(h5_address='./trained_models/AffectNet_6336.h5')
    tester.recognize_fer(img_path='./img.jpg')

```
plaese see the following [main.py](https://github.com/aliprf/Ad-Corre/tree/main/main.py) file.


## Training Network from scratch
The information and the code to train the model is provided in train.py .Plaese see the following [main.py](https://github.com/aliprf/Ad-Corre/tree/main/main.py) file:

```
    '''training part'''
    trainer = TrainModel(dataset_name=DatasetName.affectnet, ds_type=DatasetType.train_7)
    trainer.train(arch="xcp", weight_path="./")

```


### Preparing Data
Data needs to be normalized and saved in npy format. 

---------------------------------------------------------------

```
Please cite this work as:

@ARTICLE{9727163,
  author={Fard, Ali Pourramezan and Mahoor, Mohammad H.},
  journal={IEEE Access}, 
  title={Ad-Corre: Adaptive Correlation-Based Loss for Facial Expression Recognition in the Wild}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2022.3156598}}

```



