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

# Objective:
When we come across a modern-day dataset of hundreds and thousands of attributes for data...

# Variance Threshold:

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%20%28%20x_%7Bi%7D%5E%7Ba%7D%20-%20%5Cmu%5E%7Ba%7D%20%5Cright%20%29%5E%7B2%7D%7D%7Bn-1%7D)

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

**Limitation:** This method only works for numeric data.

# ANNOVA
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20F-score%20%3D%20%5Cfrac%7BBetween%20Group%20Variance%7D%7BWithin%20Group%20Variance%7D)

## Within Group variance
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20SSW%20%3D%20%5Csum_%7Bi%20%5Cepsilon%20Amazon%7D%5Cleft%20%28%20x_%7Bi%7D%20-%20%5Cmu_%7BA%7D%5Cright%20%29%5E%7B2%7D%20&plus;%20%5Csum_%7Bi%20%5Cepsilon%20Bajaj%7D%5Cleft%20%28%20x_%7Bi%7D%20-%20%5Cmu_%7BB%7D%5Cright%20%29%5E%7B2%7D%20&plus;%20%5Csum_%7Bi%20%5Cepsilon%20TCS%7D%5Cleft%20%28%20x_%7Bi%7D%20-%20%5Cmu_%7BT%7D%5Cright%20%29%5E%7B2%7D)

## Between Group Variance
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20SSB%20%3D%20n_%7B1%7D%5Cast%20%5Cleft%20%28%20%5Cmu_%7BT%7D%20-%20%5Cmu_%7Bsal%7D%20%5Cright%20%29%5E%7B2%7D%20&plus;%20n_%7B2%7D%5Cast%20%5Cleft%20%28%20%5Cmu_%7BA%7D%20-%20%5Cmu_%7Bsal%7D%20%5Cright%20%29%5E%7B2%7D%20&plus;%20n_%7B3%7D%5Cast%20%5Cleft%20%28%20%5Cmu_%7BB%7D%20-%20%5Cmu_%7Bsal%7D%20%5Cright%20%29%5E%7B2%7D)

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20F-score%20%3D%20%5Cfrac%7BSSB%20/%20DOF_%7BW%7D%7D%7BSSW/%20DOF_%7BB%7D%7D)

## Null Hypothesis 
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20H_%7B0%7D%3A%20%5Cmu_%7BA%7D%20%3D%20%5Cmu_%7BB%7D%20%3D%20%5Cmu_%7BT%7D)

# Chi-Square Test