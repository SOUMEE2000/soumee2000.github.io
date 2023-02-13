---
title: "Feature Engineering - I "
excerpt: "**Let's dig into Selecting the proper features to train our models with Statistics - i) Variance-based methods, ii) Correlation-based, iii) ANNOVA test, iv) Chi-square test.**"
tags:
     - Statistics
categories: Feature-Engineering
author_profile: false
classes: wide
header: 
   overlay_image: "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832_960_720.jpg"
   overlay_filter: 0.00
---

# Feature Selection:
Today's datasest consist of terrabytes of data. It is not feasible computationally to train our model on all these prameters. Proper training does require a lot of data, don't get me wrong, but not all of what we have maybe useful for our specific task. Neither is it memory-wise efficient to load all that data into our 16GB memory. There maybe various ways to do this programatically, like using data-loaders( PyTorch DataLoader, MONAI), but another way out can be to analyse the data statiscally and determine whether the data is statiscally significant or not. These techniques are called **Feature Selection.**

In this blog I will be mostly focussing on some of the statiscal techniques and in a future blog talk write some python code to analyse some real-world data, maybe from the [**CERN open-data platform**](https://opendata.cern.ch/search?page=1&size=20&q=).

There are broadly 3 categories of feature-selection possible. Today we look at the **Filter-based methods**.

<img src = "https://github.com/SOUMEE2000/soumee2000.github.io/blob/master/assets/images/Feature%20Engineering-filter%20based-Page-2.drawio.png">

# Variance Based:

Salary | Company |
--     | --      |
1000   | A       |
1000   | B       |
..     | ..
1000   | C       |

Let's look at these columns for a bit. There is little to no variance in salary in company A, B or C. So if I am to predict a company based on salary, the salary column gives me no useful information The formula for calculating variance is this:

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%20%28%20x_%7Bi%7D%5E%7Ba%7D%20-%20%5Cmu%5E%7Ba%7D%20%5Cright%20%29%5E%7B2%7D%7D%7Bn-1%7D)

So,

for each attribute:
     calculate the the variance
     
The trributes with less variance can be rejected.

**Limitation:** This method only works for numeric data.

# Correlation Based Filtering:
Correlation based filtering unlike the one above works with two columns. Suppose there are two columns, A and B as displayed below. If these two columns have a high value of correlation between them above a certain threshold, we can reject either attribute A or attribute B because both of these then represent the same information!

Day | A  | B
--  | --  | --
1   | 1.2% |3.1%
2   | 1.8% |4.2% 
3   |2.2%  |5.0% 
4   |1.5%  |4.2% 

## Formulae
Correlation is related to covariance. The formula to calculate covariance between two attribute columns A and B are:

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20correlation%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7B%5Cleft%20%28%20x_%7Bi%7D%5E%7Ba%7D%20-%20%5Cmu%5E%7Ba%7D%20%5Cright%20%29%5Cast%20%5Cleft%20%28%20x_%7Bi%7D%5E%7Bb%7D%20-%20%5Cmu%5E%7Bb%7D%20%5Cright%20%29%7D%7Bn-1%7D)

where uA is the mean of all the data in attribute A and same for uB. Sample calculation:

Day 1=(1.2−1.675)×(3.1−4.125)=0.487

Day 2=(1.8−1.675)∗(4.2−4.125)=0.009

Day 3=(2.2−1.675)∗(5.0−4.125)=0.459

Day 4=(1.5−1.675)∗(4.2−4.125)=−0.013

Correlation = 0.487+0.009+0.459−0.013 /4-1= 0.943/3

Now all you need is a proper threshold beyond which if the correlation goes you reject one of the attributes.

**Limitation:** This method only works for numeric data. Sad right?

Don't worry ANNOVA and Chi-square to your rescue, coming soon!
