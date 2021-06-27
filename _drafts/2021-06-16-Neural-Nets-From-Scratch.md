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

## The Math for the feedforward part

The process by which the input to a perceptron move forward through it is what is called the feedforward part of the algorithm. And that calculation above definitely looks familiar. Pretty much the same thing happens for matrix multiplication, doesn't it?

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%7B%5Ccolor%7BDarkGreen%7D%20%5Cbegin%7Bbmatrix%7D%204%20%26%20100%20%26%202%20%26%204%20%5Cend%7Bbmatrix%7D%7D%20%5Ccdot%20%7B%5Ccolor%7BRed%7D%5Cbegin%7Bbmatrix%7D%201%5C%5C%204%5C%5C%201%5C%5C%201%20%5Cend%7Bbmatrix%7D%7D%20%3D%20%7B%5Ccolor%7BBlue%7D%5Cbegin%7Bbmatrix%7D%201*4%20&plus;%204*100&plus;%201*2&plus;%201*4%20%5Cend%7Bbmatrix%7D%20%7D%20%3D%20%7B%5Ccolor%7BBlue%7D%5Cbegin%7Bbmatrix%7D%20410%20%5Cend%7Bbmatrix%7D%7D)

The red matrix has the weights and the green one contains the feature values. The python code for the above calculation is.

```
import numpy as np

X_input = np.array([4,100,2,4])
weights= np.array([1,4,1,1])
sum = np.dot(X_input, weights.T) + 0.02
```
But what is that 0.02? Well that is known as the bias term. 


![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Csum%20X_%7Binput%7D%20.%20W%5E%7BT%7D%20&plus;%20b)

And on the whole this equation should look very similar. It is of the form **Y = MX + C**. That is the function of a straight line with slope M and and intercept c!!!. Suppose we want to classify a dog and a cat only based on the features colour and size. So, on a graph each of those points would map to an (x,y) co-ordinates.  On a 2-D plane, you could imagine that the perceptron draws a line separating the red dots and blue dots ( elephants and cats). At first we start off with a random line and then we keep modifying the biases and weights until we get a decent enough line that can properly separate the red dots and blue dots. So, any new point would fall on either side of the line and the perceptron would say that this is this!!!!

<img src= "https://user-images.githubusercontent.com/52605586/123555006-a2bd8280-d7a0-11eb-9fdd-d4be12866cbd.png">

If we consider 3 features, then the plotting will be on the basis of (x,y,z). So, there will be a plane separating the two groups of points. For d features, it will be a d-dimensional feature space but obviously that is pretty hard to imagine😅.

```
def sigmoid(x):
    return 1/(1+np.exp(-x))
```

## The Math for backpropagation

**Loss function:**

```
def gradient(x):
  return x * (1-x)
```

```
 error = (Y- predicted_output)
 adjustment = error * gradient(predicted_output)   ### gradient descent
 weights += 0.01*np.dot(X.T,adjustment)     # input_samples.T is transpose so that matrix dimension matches
```
From here afterwards nothing theoretical is going to happen because I think that's enough theory for 1 blog-post😂😂😂. I will just run you down the essentials of the code you will find in this [colab notebook](https://github.com/SOUMEE2000/Machine-Learning-Stash/blob/main/Neural%20Networks/Neural_nets_From_Scratch.ipynb). Go through this first as all the standard steps for preprocessing the data is in there. Also this net gave a 95.5% accuracy on the Iris Dataset.

## Single-layer perceptron

The most important part about coding this is to be very meticulous about handling the matrix dimensions.

```
import numpy as np
import seaborn as sns

X= np.array([[1,0,0],[1,0,1],[1,0,1],[0,0,0]])
Y= np.array([[1,0],[0,0],[0,1],[0,1]])
```
* Here my X is a set of 3 features....for each of the animals I want to teach my neural net about.
* Dimensions of X :  4 * 3
* My Y is a one-hot encoded vector. 
* Dimensions of Y :  4 * 2

```
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

## Multi-layer perceptron

```
class neural_network():

  def __init__(self):
    self.weights=[]
  
  def network(self, input_layer, output_layer, learning_rate):

    predicted=[]
    input=input_layer
    for i in self.weights:
      X_input_hidden= np.dot(input, i)
      predicted_output_hidden= sigmoid(X_input_hidden)
      predicted.append(predicted_output_hidden)
      input= predicted_output_hidden

    error=[]
    error_output = predicted[-1]- output_layer
    error.append(error_output)
    self.weights.reverse()
    
    for i in self.weights:
      error_hidden= np.dot(error_output,i.T)
      error.insert(0, error_hidden)
      error_output= error_hidden

    self.weights.reverse()
    error = error[1:]
    adjustment = error[0] * gradient(predicted[0])   ### gradient descent between input-hidden layer
    self.weights[0] -= learning_rate * np.dot(input_layer.T,adjustment)

    for i in range(0,len(self.weights)-1):
      adjustment = error[i+1] * gradient(predicted[i+1])
      self.weights[i+1] -= learning_rate * np.dot(predicted[i].T, adjustment)

    return (self.weights, predicted[-1]- output_layer)

  
  
  def training_network(self, epochs, inputs, outputs, no_of_hidden_neurons, learning_rate):
    
    h= no_of_hidden_neurons
    self.weights.append(np.random.random(size=(inputs.shape[1],h[0])))
    
    for i in range(0,len(no_of_hidden_neurons)-1):
      self.weights.append(np.random.random(size=(h[i], h[i+1])))
    self.weights.append(np.random.random(size=(h[-1], outputs.shape[1])))

    for i in range(epochs):
      for j in range(len(inputs)):
        self.weights, error_output = self.network(np.array([inputs[j]]), outputs[j], learning_rate)

      print("For epoch %d error: %.3f " %(i, error_output.mean()))

    
  def predict(self, X_test):
    results=[]
    for j in X_test:
      input=j
      for i in self.weights:
        X_input_hidden= np.dot(input, i)
        predicted_output_hidden= sigmoid(X_input_hidden)
        input= predicted_output_hidden
      results.append(np.argmax(predicted_output_hidden))

    return results
```
