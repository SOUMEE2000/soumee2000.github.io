---
title: "Webscraping and Preparing a Dataset "
excerpt: "**Little Details but Important**"
tags:
     - Web Scrapping
categories: Technical
author_profile: false
classes: wide
header: 
   overlay_image: "https://images.fineartamerica.com/images/artworkimages/mediumlarge/1/enchanted-forest-jason-naudi-photography.jpg"
   overlay_filter: 0.00
---

# **Single-Layer-Perceptron**


```python
import numpy as np
import seaborn as sns
```


```python
X= np.array([[1,0,0],[1,0,1],[1,0,1],[0,0,0]])
Y= np.array([[1],[0],[0],[1]])
```


```python
X.shape[1]
```




    3




```python
#### Weights
np.random.seed(10)
weights = np.random.random((3,1))
weights
```




    array([[0.77132064],
           [0.02075195],
           [0.63364823]])




```python
weights.shape
```




    (3, 1)




```python
bias=0.02
sum= np.dot(X,weights) +bias
```


```python
#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
```


```python
predicted_output= sigmoid(sum)
predicted_output
```




    array([[0.68811483],
           [0.8061162 ],
           [0.8061162 ],
           [0.50499983]])




```python
# cost= out-pred
error = Y - predicted_output
error
```




    array([[ 0.31188517],
           [-0.8061162 ],
           [-0.8061162 ],
           [ 0.49500017]])




```python
def gradient(x):
  return x * (1-x)
```


```python
for i in range(2000):
    sum = 0.02 + np.dot(X,weights)
    predicted_output = sigmoid(sum)
    error = (Y- predicted_output)
    adjustment = error * gradient(predicted_output)   ### gradient descent
    weights += 0.01*np.dot(X.T,adjustment)     # input_samples.T is transpose so that matrix dimension matches
error
```




    array([[ 0.02314971],
           [-0.01328825],
           [-0.01328825],
           [ 0.49500017]])



# **Multi-Layer Perceptron (Using Matrices)**
*For n neurons, 1 hidden layer*


```python
learning_rate = 0.01 #@param {type:"slider", min:0, max:1, step:0.01}
no_of_hidden_neurons = 4 #@param {type:"slider", min:0, max:10, step:1}
epochs = 500 #@param {type:"slider", min:0, max:500, step:10}
```


```python
import numpy as np
import seaborn as sns
```


```python
#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
```


```python
def gradient(x):
  return x * (1-x)
```


```python
def network(input_layer, output_layer, weights1, weights2, learning_rate):
   
   X_input_hidden= np.dot(input_layer, weights1)
   predicted_output_hidden= sigmoid(X_input_hidden)
   
   X_input_output= np.dot(predicted_output_hidden, weights2)
   predicted_output= sigmoid(X_input_output)
     
   error_output= predicted_output - output_layer               ### error at the output layer
   error_hidden= np.dot(error_output,weights2.T)                ### error at the hidden layer
   
   adjustment = error_output * gradient(predicted_output)   ### gradient descent for hidden-output layer
   weights2 -= learning_rate*np.dot(predicted_output_hidden.T,adjustment)          ### weight updation of hidden-output layer.

   adjustment = error_hidden * gradient(predicted_output_hidden)   ### gradient descent between input-hidden layer
   weights1 -= learning_rate*np.dot(input_layer.T,adjustment)           ### weight updation between input-hidden layer.
   
   return (weights1, weights2, error_output)
     
```


```python
def training_network(epochs, inputs, outputs, no_of_hidden_neurons, learning_rate):

  h =no_of_hidden_neurons

  weights1=np.random.random(size=(inputs.shape[1], h))
  weights2= np.random.random(size=(h, outputs.shape[1]))
 
  for i in range(epochs):
    for j in range(len(inputs)):
      weights1, weights2, error_output = network(np.array([inputs[j]]), np.array([outputs[j]]), weights1, weights2,learning_rate)
    print("For epoch %d error: %.3f " %(i, error_output))
```


```python

dataset = np.array([[2.7810836,2.550537003],
	[1.465489372,2.362125076],
	[3.396561688,4.400293529],
	[1.38807019,1.850220317],
	[3.06407232,3.005305973],
	[7.627531214,2.759262235],
	[5.332441248,2.088626775],
	[6.922596716,1.77106367],
	[8.675418651,-0.242068655],
	[7.673756466,3.508563011]])
Y= np.array([[1],[0],[1],[0],[0],[1],[1],[0],[1],[0]])

```


```python
training_network(epochs, dataset, Y, no_of_hidden_neurons, learning_rate)
```

    For epoch 0 error: 0.867 
    For epoch 1 error: 0.865 
    For epoch 2 error: 0.863 
    For epoch 3 error: 0.862 
    For epoch 4 error: 0.860 
    For epoch 5 error: 0.858 
    For epoch 6 error: 0.856 
    For epoch 7 error: 0.854 
    For epoch 8 error: 0.852 
    For epoch 9 error: 0.850 
    For epoch 10 error: 0.848 
    For epoch 11 error: 0.846 
    For epoch 12 error: 0.844 
    For epoch 13 error: 0.842 
    For epoch 14 error: 0.840 
    For epoch 15 error: 0.838 
    For epoch 16 error: 0.836 
    For epoch 17 error: 0.834 
    For epoch 18 error: 0.832 
    For epoch 19 error: 0.830 
    For epoch 20 error: 0.828 
    For epoch 21 error: 0.826 
    For epoch 22 error: 0.823 
    For epoch 23 error: 0.821 
    For epoch 24 error: 0.819 
    For epoch 25 error: 0.817 
    For epoch 26 error: 0.814 
    For epoch 27 error: 0.812 
    For epoch 28 error: 0.810 
    For epoch 29 error: 0.808 
    For epoch 30 error: 0.805 
    For epoch 31 error: 0.803 
    For epoch 32 error: 0.800 
    For epoch 33 error: 0.798 
    For epoch 34 error: 0.796 
    For epoch 35 error: 0.793 
    For epoch 36 error: 0.791 
    For epoch 37 error: 0.788 
    For epoch 38 error: 0.786 
    For epoch 39 error: 0.784 
    For epoch 40 error: 0.781 
    For epoch 41 error: 0.779 
    For epoch 42 error: 0.776 
    For epoch 43 error: 0.774 
    For epoch 44 error: 0.771 
    For epoch 45 error: 0.769 
    For epoch 46 error: 0.766 
    For epoch 47 error: 0.763 
    For epoch 48 error: 0.761 
    For epoch 49 error: 0.758 
    For epoch 50 error: 0.756 
    For epoch 51 error: 0.753 
    For epoch 52 error: 0.751 
    For epoch 53 error: 0.748 
    For epoch 54 error: 0.746 
    For epoch 55 error: 0.743 
    For epoch 56 error: 0.740 
    For epoch 57 error: 0.738 
    For epoch 58 error: 0.735 
    For epoch 59 error: 0.733 
    For epoch 60 error: 0.730 
    For epoch 61 error: 0.728 
    For epoch 62 error: 0.725 
    For epoch 63 error: 0.722 
    For epoch 64 error: 0.720 
    For epoch 65 error: 0.717 
    For epoch 66 error: 0.715 
    For epoch 67 error: 0.712 
    For epoch 68 error: 0.710 
    For epoch 69 error: 0.707 
    For epoch 70 error: 0.705 
    For epoch 71 error: 0.702 
    For epoch 72 error: 0.700 
    For epoch 73 error: 0.697 
    For epoch 74 error: 0.695 
    For epoch 75 error: 0.692 
    For epoch 76 error: 0.690 
    For epoch 77 error: 0.687 
    For epoch 78 error: 0.685 
    For epoch 79 error: 0.683 
    For epoch 80 error: 0.680 
    For epoch 81 error: 0.678 
    For epoch 82 error: 0.676 
    For epoch 83 error: 0.673 
    For epoch 84 error: 0.671 
    For epoch 85 error: 0.669 
    For epoch 86 error: 0.666 
    For epoch 87 error: 0.664 
    For epoch 88 error: 0.662 
    For epoch 89 error: 0.660 
    For epoch 90 error: 0.657 
    For epoch 91 error: 0.655 
    For epoch 92 error: 0.653 
    For epoch 93 error: 0.651 
    For epoch 94 error: 0.649 
    For epoch 95 error: 0.647 
    For epoch 96 error: 0.644 
    For epoch 97 error: 0.642 
    For epoch 98 error: 0.640 
    For epoch 99 error: 0.638 
    For epoch 100 error: 0.636 
    For epoch 101 error: 0.634 
    For epoch 102 error: 0.632 
    For epoch 103 error: 0.630 
    For epoch 104 error: 0.629 
    For epoch 105 error: 0.627 
    For epoch 106 error: 0.625 
    For epoch 107 error: 0.623 
    For epoch 108 error: 0.621 
    For epoch 109 error: 0.619 
    For epoch 110 error: 0.618 
    For epoch 111 error: 0.616 
    For epoch 112 error: 0.614 
    For epoch 113 error: 0.612 
    For epoch 114 error: 0.611 
    For epoch 115 error: 0.609 
    For epoch 116 error: 0.607 
    For epoch 117 error: 0.606 
    For epoch 118 error: 0.604 
    For epoch 119 error: 0.603 
    For epoch 120 error: 0.601 
    For epoch 121 error: 0.599 
    For epoch 122 error: 0.598 
    For epoch 123 error: 0.596 
    For epoch 124 error: 0.595 
    For epoch 125 error: 0.594 
    For epoch 126 error: 0.592 
    For epoch 127 error: 0.591 
    For epoch 128 error: 0.589 
    For epoch 129 error: 0.588 
    For epoch 130 error: 0.587 
    For epoch 131 error: 0.585 
    For epoch 132 error: 0.584 
    For epoch 133 error: 0.583 
    For epoch 134 error: 0.581 
    For epoch 135 error: 0.580 
    For epoch 136 error: 0.579 
    For epoch 137 error: 0.578 
    For epoch 138 error: 0.577 
    For epoch 139 error: 0.575 
    For epoch 140 error: 0.574 
    For epoch 141 error: 0.573 
    For epoch 142 error: 0.572 
    For epoch 143 error: 0.571 
    For epoch 144 error: 0.570 
    For epoch 145 error: 0.569 
    For epoch 146 error: 0.568 
    For epoch 147 error: 0.567 
    For epoch 148 error: 0.566 
    For epoch 149 error: 0.565 
    For epoch 150 error: 0.564 
    For epoch 151 error: 0.563 
    For epoch 152 error: 0.562 
    For epoch 153 error: 0.561 
    For epoch 154 error: 0.560 
    For epoch 155 error: 0.559 
    For epoch 156 error: 0.558 
    For epoch 157 error: 0.558 
    For epoch 158 error: 0.557 
    For epoch 159 error: 0.556 
    For epoch 160 error: 0.555 
    For epoch 161 error: 0.554 
    For epoch 162 error: 0.554 
    For epoch 163 error: 0.553 
    For epoch 164 error: 0.552 
    For epoch 165 error: 0.551 
    For epoch 166 error: 0.551 
    For epoch 167 error: 0.550 
    For epoch 168 error: 0.549 
    For epoch 169 error: 0.548 
    For epoch 170 error: 0.548 
    For epoch 171 error: 0.547 
    For epoch 172 error: 0.546 
    For epoch 173 error: 0.546 
    For epoch 174 error: 0.545 
    For epoch 175 error: 0.545 
    For epoch 176 error: 0.544 
    For epoch 177 error: 0.543 
    For epoch 178 error: 0.543 
    For epoch 179 error: 0.542 
    For epoch 180 error: 0.542 
    For epoch 181 error: 0.541 
    For epoch 182 error: 0.541 
    For epoch 183 error: 0.540 
    For epoch 184 error: 0.540 
    For epoch 185 error: 0.539 
    For epoch 186 error: 0.539 
    For epoch 187 error: 0.538 
    For epoch 188 error: 0.538 
    For epoch 189 error: 0.537 
    For epoch 190 error: 0.537 
    For epoch 191 error: 0.536 
    For epoch 192 error: 0.536 
    For epoch 193 error: 0.535 
    For epoch 194 error: 0.535 
    For epoch 195 error: 0.535 
    For epoch 196 error: 0.534 
    For epoch 197 error: 0.534 
    For epoch 198 error: 0.533 
    For epoch 199 error: 0.533 
    For epoch 200 error: 0.533 
    For epoch 201 error: 0.532 
    For epoch 202 error: 0.532 
    For epoch 203 error: 0.532 
    For epoch 204 error: 0.531 
    For epoch 205 error: 0.531 
    For epoch 206 error: 0.531 
    For epoch 207 error: 0.530 
    For epoch 208 error: 0.530 
    For epoch 209 error: 0.530 
    For epoch 210 error: 0.529 
    For epoch 211 error: 0.529 
    For epoch 212 error: 0.529 
    For epoch 213 error: 0.528 
    For epoch 214 error: 0.528 
    For epoch 215 error: 0.528 
    For epoch 216 error: 0.528 
    For epoch 217 error: 0.527 
    For epoch 218 error: 0.527 
    For epoch 219 error: 0.527 
    For epoch 220 error: 0.527 
    For epoch 221 error: 0.526 
    For epoch 222 error: 0.526 
    For epoch 223 error: 0.526 
    For epoch 224 error: 0.526 
    For epoch 225 error: 0.525 
    For epoch 226 error: 0.525 
    For epoch 227 error: 0.525 
    For epoch 228 error: 0.525 
    For epoch 229 error: 0.525 
    For epoch 230 error: 0.524 
    For epoch 231 error: 0.524 
    For epoch 232 error: 0.524 
    For epoch 233 error: 0.524 
    For epoch 234 error: 0.524 
    For epoch 235 error: 0.523 
    For epoch 236 error: 0.523 
    For epoch 237 error: 0.523 
    For epoch 238 error: 0.523 
    For epoch 239 error: 0.523 
    For epoch 240 error: 0.523 
    For epoch 241 error: 0.522 
    For epoch 242 error: 0.522 
    For epoch 243 error: 0.522 
    For epoch 244 error: 0.522 
    For epoch 245 error: 0.522 
    For epoch 246 error: 0.522 
    For epoch 247 error: 0.522 
    For epoch 248 error: 0.521 
    For epoch 249 error: 0.521 
    For epoch 250 error: 0.521 
    For epoch 251 error: 0.521 
    For epoch 252 error: 0.521 
    For epoch 253 error: 0.521 
    For epoch 254 error: 0.521 
    For epoch 255 error: 0.521 
    For epoch 256 error: 0.520 
    For epoch 257 error: 0.520 
    For epoch 258 error: 0.520 
    For epoch 259 error: 0.520 
    For epoch 260 error: 0.520 
    For epoch 261 error: 0.520 
    For epoch 262 error: 0.520 
    For epoch 263 error: 0.520 
    For epoch 264 error: 0.520 
    For epoch 265 error: 0.520 
    For epoch 266 error: 0.519 
    For epoch 267 error: 0.519 
    For epoch 268 error: 0.519 
    For epoch 269 error: 0.519 
    For epoch 270 error: 0.519 
    For epoch 271 error: 0.519 
    For epoch 272 error: 0.519 
    For epoch 273 error: 0.519 
    For epoch 274 error: 0.519 
    For epoch 275 error: 0.519 
    For epoch 276 error: 0.519 
    For epoch 277 error: 0.519 
    For epoch 278 error: 0.519 
    For epoch 279 error: 0.518 
    For epoch 280 error: 0.518 
    For epoch 281 error: 0.518 
    For epoch 282 error: 0.518 
    For epoch 283 error: 0.518 
    For epoch 284 error: 0.518 
    For epoch 285 error: 0.518 
    For epoch 286 error: 0.518 
    For epoch 287 error: 0.518 
    For epoch 288 error: 0.518 
    For epoch 289 error: 0.518 
    For epoch 290 error: 0.518 
    For epoch 291 error: 0.518 
    For epoch 292 error: 0.518 
    For epoch 293 error: 0.518 
    For epoch 294 error: 0.518 
    For epoch 295 error: 0.518 
    For epoch 296 error: 0.518 
    For epoch 297 error: 0.517 
    For epoch 298 error: 0.517 
    For epoch 299 error: 0.517 
    For epoch 300 error: 0.517 
    For epoch 301 error: 0.517 
    For epoch 302 error: 0.517 
    For epoch 303 error: 0.517 
    For epoch 304 error: 0.517 
    For epoch 305 error: 0.517 
    For epoch 306 error: 0.517 
    For epoch 307 error: 0.517 
    For epoch 308 error: 0.517 
    For epoch 309 error: 0.517 
    For epoch 310 error: 0.517 
    For epoch 311 error: 0.517 
    For epoch 312 error: 0.517 
    For epoch 313 error: 0.517 
    For epoch 314 error: 0.517 
    For epoch 315 error: 0.517 
    For epoch 316 error: 0.517 
    For epoch 317 error: 0.517 
    For epoch 318 error: 0.517 
    For epoch 319 error: 0.517 
    For epoch 320 error: 0.517 
    For epoch 321 error: 0.517 
    For epoch 322 error: 0.517 
    For epoch 323 error: 0.517 
    For epoch 324 error: 0.517 
    For epoch 325 error: 0.517 
    For epoch 326 error: 0.517 
    For epoch 327 error: 0.517 
    For epoch 328 error: 0.517 
    For epoch 329 error: 0.517 
    For epoch 330 error: 0.517 
    For epoch 331 error: 0.517 
    For epoch 332 error: 0.517 
    For epoch 333 error: 0.517 
    For epoch 334 error: 0.517 
    For epoch 335 error: 0.517 
    For epoch 336 error: 0.517 
    For epoch 337 error: 0.517 
    For epoch 338 error: 0.517 
    For epoch 339 error: 0.516 
    For epoch 340 error: 0.516 
    For epoch 341 error: 0.516 
    For epoch 342 error: 0.516 
    For epoch 343 error: 0.516 
    For epoch 344 error: 0.516 
    For epoch 345 error: 0.516 
    For epoch 346 error: 0.516 
    For epoch 347 error: 0.516 
    For epoch 348 error: 0.516 
    For epoch 349 error: 0.516 
    For epoch 350 error: 0.516 
    For epoch 351 error: 0.516 
    For epoch 352 error: 0.516 
    For epoch 353 error: 0.516 
    For epoch 354 error: 0.516 
    For epoch 355 error: 0.516 
    For epoch 356 error: 0.516 
    For epoch 357 error: 0.516 
    For epoch 358 error: 0.516 
    For epoch 359 error: 0.516 
    For epoch 360 error: 0.516 
    For epoch 361 error: 0.516 
    For epoch 362 error: 0.516 
    For epoch 363 error: 0.516 
    For epoch 364 error: 0.516 
    For epoch 365 error: 0.516 
    For epoch 366 error: 0.516 
    For epoch 367 error: 0.516 
    For epoch 368 error: 0.516 
    For epoch 369 error: 0.516 
    For epoch 370 error: 0.516 
    For epoch 371 error: 0.516 
    For epoch 372 error: 0.516 
    For epoch 373 error: 0.516 
    For epoch 374 error: 0.516 
    For epoch 375 error: 0.516 
    For epoch 376 error: 0.517 
    For epoch 377 error: 0.517 
    For epoch 378 error: 0.517 
    For epoch 379 error: 0.517 
    For epoch 380 error: 0.517 
    For epoch 381 error: 0.517 
    For epoch 382 error: 0.517 
    For epoch 383 error: 0.517 
    For epoch 384 error: 0.517 
    For epoch 385 error: 0.517 
    For epoch 386 error: 0.517 
    For epoch 387 error: 0.517 
    For epoch 388 error: 0.517 
    For epoch 389 error: 0.517 
    For epoch 390 error: 0.517 
    For epoch 391 error: 0.517 
    For epoch 392 error: 0.517 
    For epoch 393 error: 0.517 
    For epoch 394 error: 0.517 
    For epoch 395 error: 0.517 
    For epoch 396 error: 0.517 
    For epoch 397 error: 0.517 
    For epoch 398 error: 0.517 
    For epoch 399 error: 0.517 
    For epoch 400 error: 0.517 
    For epoch 401 error: 0.517 
    For epoch 402 error: 0.517 
    For epoch 403 error: 0.517 
    For epoch 404 error: 0.517 
    For epoch 405 error: 0.517 
    For epoch 406 error: 0.517 
    For epoch 407 error: 0.517 
    For epoch 408 error: 0.517 
    For epoch 409 error: 0.517 
    For epoch 410 error: 0.517 
    For epoch 411 error: 0.517 
    For epoch 412 error: 0.517 
    For epoch 413 error: 0.517 
    For epoch 414 error: 0.517 
    For epoch 415 error: 0.517 
    For epoch 416 error: 0.517 
    For epoch 417 error: 0.517 
    For epoch 418 error: 0.517 
    For epoch 419 error: 0.517 
    For epoch 420 error: 0.517 
    For epoch 421 error: 0.517 
    For epoch 422 error: 0.517 
    For epoch 423 error: 0.517 
    For epoch 424 error: 0.517 
    For epoch 425 error: 0.517 
    For epoch 426 error: 0.517 
    For epoch 427 error: 0.517 
    For epoch 428 error: 0.517 
    For epoch 429 error: 0.517 
    For epoch 430 error: 0.517 
    For epoch 431 error: 0.517 
    For epoch 432 error: 0.517 
    For epoch 433 error: 0.517 
    For epoch 434 error: 0.517 
    For epoch 435 error: 0.518 
    For epoch 436 error: 0.518 
    For epoch 437 error: 0.518 
    For epoch 438 error: 0.518 
    For epoch 439 error: 0.518 
    For epoch 440 error: 0.518 
    For epoch 441 error: 0.518 
    For epoch 442 error: 0.518 
    For epoch 443 error: 0.518 
    For epoch 444 error: 0.518 
    For epoch 445 error: 0.518 
    For epoch 446 error: 0.518 
    For epoch 447 error: 0.518 
    For epoch 448 error: 0.518 
    For epoch 449 error: 0.518 
    For epoch 450 error: 0.518 
    For epoch 451 error: 0.518 
    For epoch 452 error: 0.518 
    For epoch 453 error: 0.518 
    For epoch 454 error: 0.518 
    For epoch 455 error: 0.518 
    For epoch 456 error: 0.518 
    For epoch 457 error: 0.518 
    For epoch 458 error: 0.518 
    For epoch 459 error: 0.518 
    For epoch 460 error: 0.518 
    For epoch 461 error: 0.518 
    For epoch 462 error: 0.518 
    For epoch 463 error: 0.518 
    For epoch 464 error: 0.518 
    For epoch 465 error: 0.518 
    For epoch 466 error: 0.518 
    For epoch 467 error: 0.518 
    For epoch 468 error: 0.518 
    For epoch 469 error: 0.518 
    For epoch 470 error: 0.518 
    For epoch 471 error: 0.519 
    For epoch 472 error: 0.519 
    For epoch 473 error: 0.519 
    For epoch 474 error: 0.519 
    For epoch 475 error: 0.519 
    For epoch 476 error: 0.519 
    For epoch 477 error: 0.519 
    For epoch 478 error: 0.519 
    For epoch 479 error: 0.519 
    For epoch 480 error: 0.519 
    For epoch 481 error: 0.519 
    For epoch 482 error: 0.519 
    For epoch 483 error: 0.519 
    For epoch 484 error: 0.519 
    For epoch 485 error: 0.519 
    For epoch 486 error: 0.519 
    For epoch 487 error: 0.519 
    For epoch 488 error: 0.519 
    For epoch 489 error: 0.519 
    For epoch 490 error: 0.519 
    For epoch 491 error: 0.519 
    For epoch 492 error: 0.519 
    For epoch 493 error: 0.519 
    For epoch 494 error: 0.519 
    For epoch 495 error: 0.519 
    For epoch 496 error: 0.519 
    For epoch 497 error: 0.519 
    For epoch 498 error: 0.519 
    For epoch 499 error: 0.519 


The error has definitely decreased.


```python
np.array([dataset[0]]).shape
```




    (1, 2)




```python
Y.shape
```




    (10, 1)



# **Multi-Layer Perceptron (Using Matrices)**

*For n neurons, m hidden layers*


```python

```


```python
!jupyter nbconvert --to markdown Neural-nets-From-Scratch.ipynb
```

    [NbConvertApp] WARNING | pattern u'Neural-nets-From-Scratch.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb) to various other
    formats.
    
    WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    -------
    
    Arguments that take values are actually convenience aliases to full
    Configurables, whose aliases are listed on the help line. For more information
    on full configurables, see '--help-all'.
    
    --execute
        Execute the notebook prior to export.
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
    --no-input
        Exclude input cells and output prompts from converted document. 
        This mode is ideal for generating code-free reports.
    --stdout
        Write notebook output to stdout instead of files.
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only 
        relevant when converting to notebook format)
    -y
        Answer yes to any questions instead of prompting.
    --clear-output
        Clear output of current file and save in place, 
        overwriting the existing notebook.
    --debug
        set log level to logging.DEBUG (maximize logging output)
    --no-prompt
        Exclude input and output prompts from converted document.
    --generate-config
        generate default config file
    --nbformat=<Enum> (NotebookExporter.nbformat_version)
        Default: 4
        Choices: [1, 2, 3, 4]
        The nbformat version to write. Use this to downgrade notebooks.
    --output-dir=<Unicode> (FilesWriter.build_directory)
        Default: ''
        Directory to write output(s) to. Defaults to output to the directory of each
        notebook. To recover previous default behaviour (outputting to the current
        working directory) use . as the flag value.
    --writer=<DottedObjectName> (NbConvertApp.writer_class)
        Default: 'FilesWriter'
        Writer class used to write the  results of the conversion
    --log-level=<Enum> (Application.log_level)
        Default: 30
        Choices: (0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')
        Set the log level by value or name.
    --reveal-prefix=<Unicode> (SlidesExporter.reveal_url_prefix)
        Default: u''
        The URL prefix for reveal.js (version 3.x). This defaults to the reveal CDN,
        but can be any url pointing to a copy  of reveal.js.
        For speaker notes to work, this must be a relative path to a local  copy of
        reveal.js: e.g., "reveal.js".
        If a relative path is given, it must be a subdirectory of the current
        directory (from which the server is run).
        See the usage documentation
        (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-
        slideshow) for more details.
    --to=<Unicode> (NbConvertApp.export_format)
        Default: 'html'
        The export format to be used, either one of the built-in formats
        ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf',
        'python', 'rst', 'script', 'slides'] or a dotted object name that represents
        the import path for an `Exporter` class
    --template=<Unicode> (TemplateExporter.template_file)
        Default: u''
        Name of the template file to use
    --output=<Unicode> (NbConvertApp.output_base)
        Default: ''
        overwrite base name use for output files. can only be used when converting
        one notebook at a time.
    --post=<DottedOrNone> (NbConvertApp.postprocessor_class)
        Default: u''
        PostProcessor class used to write the results of the conversion
    --config=<Unicode> (JupyterApp.config_file)
        Default: u''
        Full path of a config file.
    
    To see all available configurables, use `--help-all`
    
    Examples
    --------
    
        The simplest way to use nbconvert is
        
        > jupyter nbconvert mynotebook.ipynb
        
        which will convert mynotebook.ipynb to the default format (probably HTML).
        
        You can specify the export format with `--to`.
        Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides'].
        
        > jupyter nbconvert --to latex mynotebook.ipynb
        
        Both HTML and LaTeX support multiple output templates. LaTeX includes
        'base', 'article' and 'report'.  HTML includes 'basic' and 'full'. You
        can specify the flavor of the format used.
        
        > jupyter nbconvert --to html --template basic mynotebook.ipynb
        
        You can also pipe the output to stdout, rather than a file
        
        > jupyter nbconvert mynotebook.ipynb --stdout
        
        PDF is generated via latex
        
        > jupyter nbconvert mynotebook.ipynb --to pdf
        
        You can get (and serve) a Reveal.js-powered slideshow
        
        > jupyter nbconvert myslides.ipynb --to slides --post serve
        
        Multiple notebooks can be given at the command line in a couple of 
        different ways:
        
        > jupyter nbconvert notebook*.ipynb
        > jupyter nbconvert notebook1.ipynb notebook2.ipynb
        
        or you can specify the notebooks list in a config file, containing::
        
            c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
        
        > jupyter nbconvert --config mycfg.py
    


# **Multi-layer perceptron**


```python
from random import seed
from random import random
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
	print(layer)
```

    [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
    [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]



```python
# Calculate neuron activation for an input
from math import exp

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
```


```python
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
```


```python
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
```


```python
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
```


```python
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
```


```python
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
```


```python
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
```


```python
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 50, n_outputs)
for layer in network:
	print(layer)
```

    >epoch=0, lrate=0.500, error=6.350
    >epoch=1, lrate=0.500, error=5.531
    >epoch=2, lrate=0.500, error=5.221
    >epoch=3, lrate=0.500, error=4.951
    >epoch=4, lrate=0.500, error=4.519
    >epoch=5, lrate=0.500, error=4.173
    >epoch=6, lrate=0.500, error=3.835
    >epoch=7, lrate=0.500, error=3.506
    >epoch=8, lrate=0.500, error=3.192
    >epoch=9, lrate=0.500, error=2.898
    >epoch=10, lrate=0.500, error=2.626
    >epoch=11, lrate=0.500, error=2.377
    >epoch=12, lrate=0.500, error=2.153
    >epoch=13, lrate=0.500, error=1.953
    >epoch=14, lrate=0.500, error=1.774
    >epoch=15, lrate=0.500, error=1.614
    >epoch=16, lrate=0.500, error=1.472
    >epoch=17, lrate=0.500, error=1.346
    >epoch=18, lrate=0.500, error=1.233
    >epoch=19, lrate=0.500, error=1.132
    >epoch=20, lrate=0.500, error=1.042
    >epoch=21, lrate=0.500, error=0.961
    >epoch=22, lrate=0.500, error=0.887
    >epoch=23, lrate=0.500, error=0.821
    >epoch=24, lrate=0.500, error=0.761
    >epoch=25, lrate=0.500, error=0.707
    >epoch=26, lrate=0.500, error=0.658
    >epoch=27, lrate=0.500, error=0.613
    >epoch=28, lrate=0.500, error=0.573
    >epoch=29, lrate=0.500, error=0.536
    >epoch=30, lrate=0.500, error=0.503
    >epoch=31, lrate=0.500, error=0.472
    >epoch=32, lrate=0.500, error=0.445
    >epoch=33, lrate=0.500, error=0.420
    >epoch=34, lrate=0.500, error=0.397
    >epoch=35, lrate=0.500, error=0.376
    >epoch=36, lrate=0.500, error=0.356
    >epoch=37, lrate=0.500, error=0.339
    >epoch=38, lrate=0.500, error=0.322
    >epoch=39, lrate=0.500, error=0.307
    >epoch=40, lrate=0.500, error=0.293
    >epoch=41, lrate=0.500, error=0.280
    >epoch=42, lrate=0.500, error=0.268
    >epoch=43, lrate=0.500, error=0.257
    >epoch=44, lrate=0.500, error=0.247
    >epoch=45, lrate=0.500, error=0.237
    >epoch=46, lrate=0.500, error=0.228
    >epoch=47, lrate=0.500, error=0.220
    >epoch=48, lrate=0.500, error=0.212
    >epoch=49, lrate=0.500, error=0.204
    [{'weights': [-1.7642985345288205, 2.3911982455162994, 1.2874854815413224], 'output': 0.021873765664680658, 'delta': -0.0016979026144486168}, {'weights': [0.7125985572441096, -0.9461481235472482, -0.025025056138881438], 'output': 0.8821685290572923, 'delta': 0.0030620492059455036}]
    [{'weights': [3.8274124582073963, -1.1030446580371798, -1.189621801245296], 'output': 0.11214228485850042, 'delta': -0.011165602784295809}, {'weights': [-3.6412107110836507, 1.7142916208516394, 0.707834310255037], 'output': 0.8939308289847487, 'delta': 0.010057319901437437}]



```python

```
