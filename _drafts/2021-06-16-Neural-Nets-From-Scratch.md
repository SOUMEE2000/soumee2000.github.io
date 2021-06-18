---
title: "Cooking up a Neural Network "
excerpt: "**This is just a coding tutorial but hoping you take away more than just the code**"
tags:
     - Neural Network
categories: Technical
author_profile: false
classes: wide
header: 
   overlay_image: "https://wallpapercave.com/wp/wp5476518.jpg"
   overlay_filter: 0.00
---

# Neural Networks

<img src="https://github.com/SOUMEE2000/BLOG-Images/blob/main/Neural-nets/neuralnets1.png?raw=true">
          
This right above you is what is called a neural network. That's right ...thrown you right into the deep end🤭🤭🤭. Lots of lines and circles and even more lines and circles and they just keep coming at you. Now, for a moment if you could pretend in your mind's eye that's a black box, almost like a circuit board with input pins and output pins, I may be able to explain to you how great an innovation this is and also the intuition and reasonings about why we care(and we care a lot about this guy in machine learning) about something so complicated-looking anyway. Having established let's see if I could get you excited enough to even try hard coding this ;-). It's really just a bunch of simple things. Hold tight!!!!

## So why did someone want to build such a thing?

> "What we want is a machine that can learn from experience"
>    --- Alan Turing

That was 1947 when Alan Turing said this. So, for years after that scientists kept on thinking...
                                                            
*" but machines are idiots..if they could learn from experience....hmmm...what learns from experience...oh us!!!! Let's just try and replicate our brains then!!!!"*

And that was that!!. These neural networks are just very very simplified representations of what goes on in our brains and how we learn from experiences stored in our brain. A meticulous connection of millions of neurons(just like those circles) and millions of pathways (just like those lines), which fire and interact amongst themselves. It's fascinating(isn't it?) to see where the inspiration comes from and how it strikes them. We have come a long way since the 1950's with better algorithms but even with the best hardware available on earth, we are nowhere near to reproduce what you have in your 5 * 5 * 5 cc skull. There is a world of history and biology that I would have shared with you and probably I will write about them soon but I do encourage you to explore it on your own. You wouldn't be disppointed, I promise you.

## Let's get back to the big black box, shall we and see what goes inside it!!!

<img src="">
How do you describe something to a baby? Suppose you want a ten year old to distinguish between a cat and an elephant and given any breed of elephant or cat it classifies it correctly to its group. For an elephant you can say its got big ears, a trunk, a set of tusks and 4 feet. You could also describe it as something grey in colour, very tall and one with great strength. That too describes the elephant. And you would keep adding in more sets of features till the child gets some sort of an overall idea of how an elephant might look like. The same would go for the cat. And the next time the child sees a baby elephant on his way school(😂😂😂) based on what he has seen and understood previously, he. is going to come to a conclusion.

So what you do is:
* You feed him a set of features called **X_input** and provide a label to those features known as **X_output**
* Then you give him a new set of festures called **X_input** and you want his opinion on the **X_output**
* Mathematically, what you want to say is **P( X_output | X_input) = P( X_output ∩ X_input)∕ P( X_input | X_output)** 

Now suppose you yourself made him remember bad data or he only caught part of what you said. So what he came away from that conversation is something as heneric as a cat has 4 feet. The next time the baby encounters them if he labels an elephant as one, that's great. But the odds are he will make some mistake and as the complexity of the classification increases (classifying a healthy leaf from an unhealthy one) the chances of error due to such personal biases to one only set of parameters will be pretty costly.

So what we do....? A sensible approach would be to try minimise the error instead of giving the baby up for good. So, do we just reiterate over everything we just said to him???
I am afraid yes. But maybe give him a bit of time grow up and stuff and be a bit more "mature about his perspectives".

!story needs modification to drive point home

Now the story that I read to you right now can be extrapolated to how a neural network approaches the process of learning. That's the intuition right there. It's because we want to code our own neural network that we need to define this story a little more mathematically.

## Defining 

