---
title: "Feature Engineering - I "
excerpt: "**Let's dig into Selecting the proper features to train our models with a little bit of Statistics - i) Variance-based methods, ii) Correlation-based, iii) ANNOVA test, iv) Chi-square test.**"
tags:
     - Neural Network
categories: Feature-Engineering
author_profile: false
classes: wide
header: 
   overlay_image: "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832_960_720.jpg"
   overlay_filter: 0.00
---

# Objective:
Yo

# Variance Threshold:
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%20%28%20x_%7Bi%7D%5E%7Ba%7D%20-%20%5Cmu%5E%7Ba%7D%20%5Cright%20%29%5E%7B2%7D%7D%7Bn-1%7D)

# Correlation Based Filtering:
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20correlation%20%3D%20%5Cfrac%7B%5Cleft%20%28%20x_%7Bi%7D%5E%7Ba%7D%20-%20%5Cmu%5E%7Ba%7D%20%5Cright%20%29%5Cast%20%5Cleft%20%28%20x_%7Bi%7D%5E%7Bb%7D%20-%20%5Cmu%5E%7Bb%7D%20%5Cright%20%29%7D%7Bn-1%7D)

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
