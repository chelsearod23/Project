## Tuberculosis Classification in Chest X-ray Images using DenseNet and Deep Forest Models

#### Author

Chelsea Rodrigues - 23200333

#### Introduction

Tuberculosis is caused by a bacterium called Mycobacterium, which is contagious and spreads through the community.TB is one of the leading causes of death throughout the world. Every year, millions of people become ill with tuberculosis (TB). In 2018, approximately ten million people were infected with tuberculosis, and a million and a half people died as a result of the infection. The disease's impact varies greatly, from fewer than 5 to more than 500 new patients per 100 000 citizens each year, depending on the severity of the disease. The fundamental reason for this high death rate is a failure in tuberculosis detection: more than one-third of the approximately ten million tuberculosis incidents are not registered and detected. There were a slew of solutions developed.

Chest X-Ray is one of these procedures (CXR) Chest X-rays (CXRs) are the preferred method of TB diagnosis for most early cases, owing to the low radiation dose, low cost, wide availability distinguish it from other TB detection methods. Researchers have been working on developing a computer-aided detection (CAD) system to aid in the preliminary diagnosis of tuberculosis-related diseases using medical imaging for several decades. Convolutional neural networks (CNNs) have consistently outperformed other traditional recognition algorithms in terms of achieving superior performance for image-based classification and recognition problems, due to advancements in deep learning.

#### Project Overview

Tuberculosis (TB) is an infectious disease that primarily affects the lungs and is responsible for the deaths of millions of people worldwide each year. Early detection of TB is crucial in combating its spread and improving patient outcomes. As powerful models for extracting informative features from images, Convolutional Neural Networks (CNNs) have demonstrated advantages in medical image recognition applications as well as other fields.. It is critical in this modern age to use X-rays to detect and classify diseases affecting the chest because of the limited number of qualified radiologists available in this field. We will look at the classification of tuberculosis in chest x-ray images using DenseNet and Deep forest models.

#### Motivation

The limited availability of qualified radiologists poses a significant challenge in diagnosing chest-related diseases, including TB. Automating this process using deep learning and image processing techniques can alleviate the strain on healthcare systems and improve diagnostic accuracy.

#### Methodology

⦁ Datasets In this database, you will find X-ray images of the chest taken from Tuberculosis (TB) positive cases as well as normal images. At the time of this release, we have 700 TB images that are publicly accessible and 2800 TB images that can be downloaded from the NIAID TB portal after signing an agreement, in addition to 3500 standard images.

![](images/Screenshot%20(51)-01.png){width="583"}

Fig: Dataset Sample

#### Data Preparation:

It is necessary to undertake pre-processing in order for machine learning to be performed in accordance with medical guidelines. This includes data cleaning and normalization as well as noisy data filtering and the handling of missing values. It is vital to note that data pre-processing has a significant impact on the performance of machine-learning algorithms, and if it is not done correctly, it might result in biased output.

⦁ Preprocessing: All CXR images have been enhanced using contrast limited adaptive histogram equalization (CLAHE) with a clip limit number equal to 1.25; in addition, all CXR images have been resized from their original sizes to 150 x 150 pixels in Deep Forest model and 300 x 300 pixels in DenseNet model in order to improve processing speed.

⦁ Data augmentation: In order to increase the number of images under the classes with fewer CXRs, techniques such as horizontal flipping, rotating, contrast adjustment, and position translation have been judiciously implemented, resulting in a more evenly distributed data set that eliminates interference.

#### Machine Learning Models:

To diagnose tuberculosis and localize it in CXR images, deep CNN models are being used in conjunction with unified approaches to improve the accuracy-stability of the disease detection process. These approaches include DenseNet and cascaded classifier-based Deep forest models, among others. Using a dataset that has been divided into testing groups, we examine the overall performance of the models under consideration.

![](images/Screenshot%20(50)-01.png){width="565"}

#### 1.Dense Net

DenseNet is a convolutional neural network in which each layer is connected to all other layers that are deeper in the network. This is done in order to ensure that the most amount of information can flow between the layers of the network. In order to maintain the feed-forward nature of the system, each layer obtains inputs from all of the previous layers and passes on its own feature maps to all of the layers that will come after it.

Dense Net is better then others because in other models each layer passes information only to next layer which might lead to information loss but in dense every layer gets information from all previous layers so it has full picture which leads to better understanding of complex things. Also Dense net focuses only on important features and ignores unnecessary things due to which less memory is used and is more efficient than other models. It better understand patterns that indicate presence of TB

Our investigation focuses on the binary classification of images in order to distinguish between TB abnormalities, while also performing a further diagnosis and localization of specific TB-related manifestations on the data set under consideration.

![](images/Screenshot%20(52)-01.png){width="476"}

Fig: DenseNet Architecture

#### 2. Deep Forest

Deep forest is a ensemble learning ,method which means it is an machine learning approach where multiple models are combined to make predictions which improves overall accuracy and robustness of prediction compared to single model. It predicts bounding boxes around areas of interest in chest xray to identify where TB can be. It utilizes Cascaded structure that is each stage refines prediction made by previous stage and performs well in classifying CXR images without segmentation. An ensemble of learners has long been recognised as having great generalization performance as compared to solo learners.

![](images/Screenshot%20(53)-01.png){width="529"}

Fig : Deep Forest Architecture

#### Prerequisites

-   Python 3.x
-   TensorFlow
-   Scikit-learn
-   OpenCV
-   Numpy
-   Pandas
-   Matplotlib
-   deepforest

#### How to Run the Project

1.  Clone the repository:

```         
git clone <repository-url> cd <repository-folder>
```

2.  Install the required packages:

```         
pip install -r requirements.txt
```

3.  Run the project script:

```         
jupyter notebook
```

4.  Open and run the notebooks in the Jupyter Notebook interface.

#### Execution Time

The entire project, including training and evaluation, is expected to run in approximately 20 minutes.

#### Results

In this two-class problem, all of the evaluated pre-trained models perform exceptionally well in distinguishing between TB and normal images. Deep Forest outperforms the other networks trained with X-ray images without segmentation when it comes to classifying the X-ray images, according to the results.

![](images/Screenshot%20(55)-01.png){width="461"}

![](images/Screenshot%20(56)-01.png){width="392"}

Fig: Deep forest Predicted label vs True label

#### ![](images/Screenshot%20(57).png){width="376"}

![](images/Screenshot%20(58).png){width="366"}

Fig : Dense Net Accuracy and Loss Plot

![](images/Screenshot%20(59).png){width="344"}

Fig:DenseNet Predicted label Vs True label

![](images/Screenshot%20(60).png){width="390"}

#### License

This project is licensed under the MIT License - see the LICENSE file for details.

#### Conclusion and future work

For the classification of TB and normal CXR images, the performance of deep forest and CNN models was evaluated. According to the Deep forest method, classification accuracy, precision, and recall for the detection of tuberculosis were found to be 97 percent, 97 percent, and 97 percent, respectively, while the DenseNet method achieved 50 percent, 25 percent, and 33 percent. For this reason, segmentation of the lungs is extremely important when performing a computer-aided diagnosis using radiographs. This cutting-edge performance has the potential to be a very useful and quick diagnostic tool, which has the potential to save a significant number of people who die every year as a result of delayed or improper diagnosis. When we use datasets without image segmentation, the DenseNet201 model do not performs well.

#### References

C. Liu, M. Alcantara and B. Liu, "TX-CNN: Detecting tuberculosis in chest X-ray images using convolutional neural network", IEEE, 2017. [Accessed 31 July 2020].

D. Jeoung, O. Stephen and M. Sain, "An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare", Journal of Healthcare Engineering, vol. 2019, 2019. [Accessed 2 August 2020]

Anthimopoulos, M., Christodoulidis, S., Ebner, L., Christe, A., and Mougiakakou, S. (2016). Lung pattern classification for interstitial lung diseases using a deep convolutional neural network. IEEE Trans. Med. Imaging 35, 1207–1216. doi: 10.1109/TMI.2016.2535865

Boureau, Y. L., Le Roux, N., Bach, F., Ponce, J., and LeCun, Y. (2011). “Ask the locals: multi-way local pooling for image recognition,” in ICCV'11-The 13th International Conference on Computer Vision (Barcelona). doi: 10.1109/ICCV.2011.6126555

Pasa, F.; Golkov, V.; Pfeiffer, F.; Cremers, D.; Pfeiffer, D. Efﬁcient Deep Network Architectures for Fast ChestX-Ray Tuberculosis Screening and Visualization. Sci. Rep. 2019,9, 6268

Anderson, L.; Dean, A.; Falzon, D.; Floyd, K.; Baena, I.; Gilpin, C.; Glaziou, P.; Hamada, Y.; Hiatt, T.; Char, A.;et al. Global tuberculosis report 2015. WHO Libr. Cat. Data 2015,1, 1689–1699

NASH M., KADAVIGERE R., ANDRADE J, et al. Deep learning, computer-aided radiography reading for tuberculosis: a diagnostic accuracy study from a tertiary hospital in India. Scientific Reports, 2020, 10(1):1-10.

DUONG L. T., LE N. H., TRAN T. B, et al. Detection of Tuberculosis from Chest X-ray Images: Boosting the Performance with Vision Transformer and Transfer Learning. Expert Systems with Applications, 2021:115519.

DASANAYAKA C., and DISSANAYAKE M. B. Deep Learning Methods for Screening Pulmonary Tuberculosis Using Chest X-rays. Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization, 2020:1-11.

YADAV S. S., and JADHAV S. M. Deep convolutional neural network based medical image classification for disease diagnosis. Journal of Big Data, 2019, 6(1):1-18.

KARNKAWINPONG T., and LIMPIYAKORN Y. Classification of pulmonary tuberculosis lesion with convolutional neural networks. Journal of Physics: Conference Series, 2019, 1195: 012007
