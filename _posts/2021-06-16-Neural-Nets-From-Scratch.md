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

## Story Time!!!

<img src="https://user-images.githubusercontent.com/52605586/123501617-d34bd200-d663-11eb-8a48-486c6c3779fd.png">
How do you describe something to a baby? Suppose you want a ten year old to distinguish between a cat and an elephant and given any breed of elephant or cat you want him to classify the creature correctly to its group. For an elephant you can say its got big ears, a trunk, a set of tusks and 4 feet. You could also describe it as something grey in colour, very tall and one with great strength. That too describes the elephant. You could also the describe to him the features of different breeds of elephants. And you would probably keep adding in more sets of features till the child gets some sort of an overall idea of how an elephant might look like. The same would go for the cat. And the next time the child sees a baby elephant on his way to school(😂😂😂), based on what he has seen and understood previously, he is going to come to a conclusion.

So what you do is:
* You feed him a set of features called **X_input** and provide a label to those features known as **X_output**
* Then you give him a new set of features called **X_input** and you want his opinion on the **X_output**
* Mathematically, what you want to find out is **X_output - X_predicted_output** which is the error ( let's not expect the child to know everything at the first shot...learning is an **iterative** process)

That error will probably come in because he couldn't remember what was told(thinking of your cooking perhaps?). So, the next time if he labels an elephant as one(error = 0), that's great. But the odds are he will make some mistakes and as the complexity of the classification increases (classifying a healthy leaf from an unhealthy one) the chances of error due to such personal biases to one only set of parameters will be pretty costly.

One objective way to do so would be to reduce the error by trying to look at how he approached the problem, trying to tinker with his thoughts for a bit and in general trying to fix his biases towards particular features. Ask him to go over those features once again fix his errors and then proceed to do so for all the sets of features you have. Now, you realise perhaps that this is hardly the best way to make a child learn (more on that later) but that's how a neural network does it. You can understand almost any algorithm if you abstract it to a real-life situation as this and this may seem very primitive but it is the cornerstone for many great ideas. 

To summarise, the algorithm is:
* You feed him a set of features called **X_input** and provide a label to those features known as **X_output**
* Then you give him a new set of features called **X_input** and you want his opinion on the **X_output**
* Mathematically, what you want to find out is **X_output - X_predicted_output** which is the error
* Teach the child by looking at what error he makes and then back to the 1st step
* Repeat it for all the sets of features possible.

The goal here is to know enough about the algorithm to be able to implement it and that's what we shall proceed to do. Do brace yourselves for a lot of maths.

## The Ground-work

Let's segment out the the basic unit of the neural net and see what exactly does it do. It's this one right here. Also known as a **perceptron**. What it does is take a vector of inputs into it. Now, what might that vector be. Relating it to the previous example, it's the vector of features. 

<img src="https://user-images.githubusercontent.com/52605586/123503141-9802d080-d66e-11eb-99df-69be1dd35716.png">

**X_input = [ 4, large, tusks, teeth ]**

But this sort of a representation the computer can not understand. What it can understand is numbers. So, suppose we make a table like this:


| Number of feet   | Size   |  Colour  |  Tusk present |
| :-----------:    | :-----:|  :----:  |   :-------:
| 4                | 100    |  2       |  4            |


Here 2 in the colour column means brown and 1 means grey, like the indices we have in maps. So our X_input from the previous example is **[ 4, 100, 2, 4]**. So, we feed it into the perceptron. Let's think a bit logically now. We have 4 numbers with us. The plan is that we do some operation on these numbers to get a single value. If that value is above a threshold value then the features correspond to a cat or they are the features of an elephant. So, what operation can make that happen? Addition? It would make sense. The size of a cat would have to be realistically 1 if the size of an elephant is 100. That in itself could create a very big segregation if we are adding things up as the sum for a cat would avrage around 10-20 while that for an elephant would be 100-115. Similarly, think about other operations that we could do with these numbers and that will be your very own algorithm!!!!!

But let's pause for a bit. What if we have a baby elephant, a newborn one. He would have a size of 20 perhaps. Suppose for a minute that he was born without or tusks. It is indeed a sorry state of affairs(only a toy example) but then his number would add up to within 25. But that is closer to a cat than to an elephant. On the other hand we can assuredly say that a cat never grows to anything more than size 10. So, maybe the deciding factor here was not the other features but its size.

Just like in our everyday lives some jobs have more priority than others, the size of the animal should have a higher priority than the more generic features such as the number of feet. That is because the size has a grater contribution to the final decision than the other factors. Ponder about it for a bit!!!!. But one way we can assign "priorities" is multiplying weights to these i.e., we take the weighted sum of 1* 4 + 4* 100 + 1* 2 + 1* 4. So, of course the threshold is going to change but the decisions that we are going to make should have more real-life significance. Similar to this some factors also should have lesser weights than what we have chosen right now. You can probably think of countless examples in which you would want to modify those weights in a certain way. 

Generally, a perceptron does so by looking at the error it has produced and modifying its priorities accordingly, something called **Backpropagation**. 
<img src="">

## Single-layer perceptron

```
import numpy as np
import seaborn as sns

X= np.array([[1,0,0],[1,0,1],[1,0,1],[0,0,0]])
Y= np.array([[1],[0],[0],[1]])

#### Weights
np.random.seed(10)
weights = np.random.random((3,1))

#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
for i in range(2000):
    sum = 0.02 + np.dot(X,weights)
    predicted_output = sigmoid(sum)
    
    # Backprop algorithm
    error = (Y- predicted_output)
    adjustment = error * gradient(predicted_output)   ### gradient descent
    weights += 0.01*np.dot(X.T,adjustment)     # input_samples.T is transpose so that matrix dimension matches
```
