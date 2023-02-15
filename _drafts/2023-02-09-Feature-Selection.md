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

<img src="https://github.com/SOUMEE2000/soumee2000.github.io/blob/master/assets/images/Feature%20Engineering-filter%20based-Page-2.drawio.png?raw=true">

# Filter-based Techniques
## Variance Based:

Salary | Company |
--     | --      |
1000   | A       |
1000   | B       |
..     | ..
1000   | C       |

Let's look at these columns for a bit. There is little to no variance in salary in company A, B or C. So if I am to predict a company based on salary, the salary column gives me no useful information The formula for calculating variance is this:

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%20%28%20x_%7Bi%7D%5E%7Ba%7D%20-%20%5Cmu%5E%7Ba%7D%20%5Cright%20%29%5E%7B2%7D%7D%7Bn-1%7D)

So,
```
for each attribute:
     calculate the variance
```     
The atrributes with less variance can be rejected.

**Limitation:** This method only works for numeric data.

## Correlation Based Filtering:
Correlation based filtering unlike the one above works with two columns. Suppose there are two columns, A and B as displayed below. If these two columns have a high value of correlation between them above a certain threshold, we can reject either attribute A or attribute B because both of these then represent the same information!

Day | A  | B
--  | --  | --
1   | 1.2% |3.1%
2   | 1.8% |4.2% 
3   |2.2%  |5.0% 
4   |1.5%  |4.2% 

### Formulae
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

Don't worry ANNOVA and Chi-square are coming to your rescue!

# ANNOVA

Companies( Non-numeric) | NumericData( Salary)
--  | --  
A                       | - 
B                       | - 
B                       | -
C                       | -

Suppose we are given this dataset and we want to know if the company column affects the Salary column in any way. If it doesn't we can remove the company column altogether because the salary of an individual doesn't seem to be affected by the company. Then we want to calculate the mean and variance of salaries for each company. In this regard, we use a metric called **F-score**. A rough calculation of this metric is given by this formula.

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20F-score%20%3D%20%5Cfrac%7BBetween%20Group%20Variance%7D%7BWithin%20Group%20Variance%7D)

So, if this F-score value is high then we feel that the company does have an impact on salary. Intuitively this makes sense. We want a low variance of salaries within the group ie., given a company people earn similar salaries. Thus the denominator(within group variance is less). And the variance between two companies need to be a lot ie., changing companies produces different salaries so the variance between the group or the numerator needs to be very high. So, overall F-score is high.

Let's consider the 3 companies Amazon, TCS and Bajaj with n1 data-points for TCS, n2 data-points for Amazon and n3 data-points for Bajaj and calculate the within group variance and the between group variance.

**Within Group variance Formulae:**

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20SSW%20%3D%20%5Csum_%7Bi%20%5Cepsilon%20Amazon%7D%5Cleft%20%28%20x_%7Bi%7D%20-%20%5Cmu_%7BA%7D%5Cright%20%29%5E%7B2%7D%20&plus;%20%5Csum_%7Bi%20%5Cepsilon%20Bajaj%7D%5Cleft%20%28%20x_%7Bi%7D%20-%20%5Cmu_%7BB%7D%5Cright%20%29%5E%7B2%7D%20&plus;%20%5Csum_%7Bi%20%5Cepsilon%20TCS%7D%5Cleft%20%28%20x_%7Bi%7D%20-%20%5Cmu_%7BT%7D%5Cright%20%29%5E%7B2%7D)

**Between Group Variance Formulae:**

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20SSB%20%3D%20n_%7B1%7D%5Cast%20%5Cleft%20%28%20%5Cmu_%7BT%7D%20-%20%5Cmu_%7Bsal%7D%20%5Cright%20%29%5E%7B2%7D%20&plus;%20n_%7B2%7D%5Cast%20%5Cleft%20%28%20%5Cmu_%7BA%7D%20-%20%5Cmu_%7Bsal%7D%20%5Cright%20%29%5E%7B2%7D%20&plus;%20n_%7B3%7D%5Cast%20%5Cleft%20%28%20%5Cmu_%7BB%7D%20-%20%5Cmu_%7Bsal%7D%20%5Cright%20%29%5E%7B2%7D)

This is the real formula for the F-score. 

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20F-score%20%3D%20%5Cfrac%7BSSB%20/%20DOF_%7BB%7D%7D%7BSSW/%20DOF_%7BW%7D%7D)

Here the variances are normalised by the degrees of freedom. DOFw here represents the Degree of freedom within the the group. That calculation is actually the number of training data points -  the number of groups/classes. DOFb here represents the Degree of freedom between the groups and that is calculated by number of groups - 1. So for X confidence interval and the two degrees of freedom we have a fixed F-score table that we can look up online. If the F-score is greater than the value obtained from the table, then we say that company name does have an impact on salary or these two values are correlated. So we can remove either the company name or the salary.

## Null Hypothesis 
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20H_%7B0%7D%3A%20%5Cmu_%7BA%7D%20%3D%20%5Cmu_%7BB%7D%20%3D%20%5Cmu_%7BT%7D)

# Chi-Square Test
