---
title: "Evolution of U-Nets"
description: "A brief walk-through of U-Nets and how they evolved to include concepts such as attention. One of the best ways to learn about something is to see how it was created and how it evolved."
pubDatetime: 2021-05-05T00:00:00+05:30
draft: true
tags:
  - Literature Review
  - U-Nets
---

# Hello There!

**U-Nets** were the first thing I learnt about when I decided to give medical image processing a shot. They are foundational models in this field, and many newer models (we will discover them eventually) use this structure as a backbone and try to improve upon it. Just in case your eyes haven't already wandered down to the picture, this is the architecture we plan to study.

## U-Net

![U-Net architecture](/assets/blog/unet/u-net.png)

The nomenclature, I believe, should be fairly obvious. But if you are new to it, your mind should be abuzz with questions!

**To name a few:**

- Why U?
- How U?
- What exactly does the U mean?

## Residual U-Net

![Residual U-Net architecture](/assets/blog/unet/residual-u-net.png)

## BCDU-Net

![BCDU-Net architecture](/assets/blog/unet/bcdu-net.png)

## DU-Net

![DU-Net architecture](/assets/blog/unet/du-net.png)

## Spatial Attention U-Net

![Spatial Attention U-Net architecture](/assets/blog/unet/spatial-attention-u-net.png)

Coding them is pretty challenging at first, not going to deny it. Maybe that's something I will do for a later blog post, but until then!

## References

- [Spatial Attention U-Net](https://arxiv.org/ftp/arxiv/papers/2004/2004.03696.pdf)
- [DU-Net](https://arxiv.org/pdf/1811.01206v1.pdf)
- [BCDU-Net](https://arxiv.org/pdf/1909.00166v1.pdf)
- [Residual U-Net](https://arxiv.org/pdf/1711.10684v1.pdf)
- [U-Net](https://arxiv.org/pdf/1505.04597v1.pdf)
