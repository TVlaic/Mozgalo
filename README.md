# Mozgalo

Mozgalo is the first and the only Croatian competition for students in Data mining and Big Data.

Through their solutions, students gather useful knowledge and information among big amounts of structured and unstructured data. They use different statistical techniques, machine learning and computer vision algorithms. One of Mozgalo’s primary goals is to encourage students to do analytical thinking, prediction modeling and independently develop their own creative solutions to real problems using their knowledge of computer science, statistics and mathematics. In addition to that, students develop their communication and business skills, teamwork and they have an opportunity to connect with companies from IT and bank sector. During the competition, educational workshops are being held and competitors are offered mentorship and online educational content. The competition is opened to all students, in teams of 2 to 4 people.

Data mining and predictive models are foundations of successful business in multiple industries and therefore this area can be considered the profession of the future.

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

More information about the task can be found in this [pdf](Microblink-task-eng.pdf)

## Table of contents

<a href="#Req">Requirements</a><br>
<a href="#Data">Getting the dataset</a><br>
<a href='#Results'>Example results</a><br>
<a href="#Getting started">Getting started</a><br>
<a href="#References">References</a><br>
<a href="#Acknowledgements">Acknowledgements</a><br>


## Requirements
<a id='Req'></a>

```
Linux
Python 3 with installed virtualenv
Nvidia GPU with installed Cuda 8 and CudaNN 9
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


## Results
<a id='Results'></a>

All networks listed bellow are trained on 70% split of training data created with a fixed seed of 42 which translates into 31500 samples.

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

## References
<a id='References'></a>


## Acknowledgements
<a id='Acknowledgements'></a>

Authors of this repo: Nikola Mrzljak, Toni Vlaić
