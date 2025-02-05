# Mozgalo

Mozgalo is the first and the only Croatian competition for students in Data mining and Big Data.

Through their solutions, students gather useful knowledge and information among big amounts of structured and unstructured data. They use different statistical techniques, machine learning and computer vision algorithms. One of Mozgalo’s primary goals is to encourage students to do analytical thinking, prediction modeling and independently develop their own creative solutions to real problems using their knowledge of computer science, statistics and mathematics. In addition to that, students develop their communication and business skills, teamwork and they have an opportunity to connect with companies from IT and bank sector. During the competition, educational workshops are being held and competitors are offered mentorship and online educational content. The competition is opened to all students, in teams of 2 to 4 people.

Data mining and predictive models are foundations of successful business in multiple industries and therefore this area can be considered the profession of the future.

More information about the competition can be found [here](https://www.estudent.hr/category/natjecanja/mozgalo-en/).

## Project theme: Robust ML Challenge

In the last couple of years, machine learning has in a way revolutionized the field of
computer vision. On some of the publicly available datasets, results surpassing
human abilities have been achieved which indicates a high potential for serious
industrial applications. On the other hand, the development of concrete solutions in
practice is often significantly more complex due to additional requirements and
limitations. The aim of this year’s Mozgalo competition is to introduce students with
the challenges of developing robust machine learning solutions. In theory, the
assignment is a simple problem of image classification which becomes highly
complex in practice. More specifically, the task is image classification of shopping
mall receipts based on their visible logo characteristics. The teams shall develop the
solution on a labelled set of 45 000 images and 25 different classes and their
robustness will be measured on the set of images which simulate the real conditions
of industrial applications

More information about the task can be found in this [pdf](Microblink-task-eng.pdf).

## Table of contents

<a href="#Req">Requirements</a><br>
<a href="#Data">Getting the dataset</a><br>
<a href='#Results'>Results</a><br>
<a href="#Getting started">Getting started</a><br>
<a href="#References">References</a><br>
<a href="#Acknowledgements">Acknowledgements</a><br>


## Requirements
<a id='Req'></a>

```
Linux
Python 3 with installed virtualenv
Nvidia GPU with installed Cuda 9
```

## Getting the dataset
<a id='Data'></a>

Since the dataset is the property of Microblink we are not allowed to publicly share it, but here are some examples of what it looks like. 

<table  border="0" width="100%" style="border:none">
<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
HEB receipt example:
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/1.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Meijer receipt example:
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/2.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Smiths receipt example
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/3.jpg" width="400">
</td>
</tr>
</table>

**Note** - _although the dataset is not available, the code is not dependant on the dataset and can be run on any classification task, as long as the images are arranged in the folder structure explained below._

## Results
<a id='Results'></a>

All networks listed bellow are trained on 70% split of training data created with a fixed seed of 42 which translates into 31500 samples. Residual Attention Networks are from modules with _small_ in their name and have a size of approximately 5 Mb.

| Network  | F1 - test score |
| ------------- | ------------- |
| Base convolutional network | ~ 0.68 |
| Residual Attention Network  | ~ 0.85  |
| Residual Attention Network with Center Loss  | ~ 0.91  |
| Ensemble of Residual Attention Networks  | ~ 0.94  |

Here are some examples of what the attention module really looks at:


<table  border="0" width="100%" style="border:none">
<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Fry's attention example:
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/att1.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
Wallgreens attention example:
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/att2.jpg" width="400">
</td>
</tr>
  
  
<tr width="100%" border="0" style="border:none">
<td border="0" align="center" style="border:none">
Jewel-Osco attention example:
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/att3.jpg" width="400">
</td>
<td border="0"  align="center" style="border:none">
WinCo Foods attention example:
<img src="https://github.com/Mungosin/Mozgalo/blob/master/ReadmeImages/att4.jpg" width="400">
</td>
</tr>
  
</table>

## Getting started
<a id='Getting started'></a>

Project is very easy to set up. Once you have met all the requirements listed above installing all the prerequisites is done by running the following command in your terminal:

```
source init.sh
```

The script will create the folder directory, create a virtual environment and install all the required libraries.

The project structure looks like this:

    .    
    ├── scripts             # root folder for all py and ipynb files
    │   ├── models          # contains model implementation
    │       ├── ..
    │   ├── preprocessors   # contains preprocessor implementations
    │       ├── ..
    │   ├── train.py        # used for model training
    │   ├── test.py         # used for model testing
    │   └── config.cfg      # model and preprocessor configuration file
    ├── inputs              # root folder for train and test directories
    │   ├── train           # directory with class directories for training
    │       ├── class1      # contains images of class 1
    │          ├── ..
    │       ├── class2      # contains images of class 2
    │          ├── ..
    │   ├── test            # directory with images just for testing
    │       ├── ..
    ├── checkpoints         # folder where all model checkpoints and tensorboard logs will be saved
    │   ├── ..              # hierarchy ./modelname/preprocessorname/date_of_experiment/checkpoint.h5
    ├── outputs             # folder where all model outputs will be saved if it has any
    │   ├── ..              # hierarchy ./modelname/preprocessorname/date_of_experiment/checkpoint.h5
    ├── init.sh             # bash initialization script
    ├── requirements.txt    # list of required packages
    └── ...

After everything is set up change the _[Data]_ properties of inputs depending on where you placed your training data. When you configured the paths in config.cfg you can start training your first model by opening a terminal in the scripts folder and running the following command:

```
python train.py [ModelName] [PreprocessorName]
```

Where ModelName could be *ResidualAttentionNet* and PreprocessorName could be *MicroblinkBasePreprocessor* and the command would look like:


```
python train.py ResidualAttentionNet MicroblinkBasePreprocessor
```

After the training process has finished you can test the last checkpoint by running the test.py script as follows:

```
python test.py ResidualAttentionNet MicroblinkBasePreprocessor
```

or if you want to run a specific checkpoint you can run it like this:

```
python test.py ResidualAttentionNet MicroblinkBasePreprocessor -p [PathToCheckpointFile]
```

## References
<a id='References'></a>

_[1]_ Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xi- aogang Wang, and Xiaoou Tang. Residual attention network for image classification. _CoRR, abs/1704.06904_, 2017. URL http://arxiv.org/abs/1704.06904.

_[2]_ Ce Qi and Fei Su. Contrastive-center loss for deep neural networks. _CoRR, abs/1707.07391_, 2017. URL http://arxiv.org/abs/1707.07391.

## Acknowledgements
<a id='Acknowledgements'></a>

Authors of this repo: Toni Vlaić, Nikola Mrzljak
