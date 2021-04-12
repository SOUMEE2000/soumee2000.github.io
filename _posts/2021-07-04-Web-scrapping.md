---
title: "Webscraping and Getting a Dataset to Kaggle Part-1 "
excerpt_separator: "<!--more-->"
tags:
     - Web Scrapping
     - Kaggle
categories: Technical
author_profile: false
classes: wide
---


##  Web-Scrapping

Now, if we are about to get started on the subject let's just get our objectives straight. Web-scrapping simply means to scrap the contents of a web-page by looking at its html(more on that later). But why is such a simple and unrelated thing mentioned in a blog-site for machine learning? The answer is simple.

Machine learning is all about data and it can be sometimes immensely helpful to be able to gather data and create one's own datasets for testing the model or just adding data to it in general. But then, why write the code when we can just copy and paste things? In fact, the example today is simple enough to make you want to do that. But, it's not possible for bigger datasets with billions of columns. So, without delay, let's get started.

This time we take the help of two python libraries:
* bs4(for web-scrapping)
* requests(for fetching the webpage from the site)

Both the libraries can be installed using:

<!--more-->
```yaml
pip install bs4
pip install requests
```

And we are good to go!!!!

Import the libraries using:

```yaml
from bs4 import BeautifulSoup
import requests
```

Let's get started.

At first we mention the url of the web-page we want to visit and call the requests library to fetch the contents of the web-page for us. The page variable stores the response.
```yaml
url="https://blackadderquotes.com/blackadder-series-4-episode-1-captain-cook-full-script"
page= requests.get(url)
page 
```

This should give a response of status 200 meaning that everything has gone on well and without any sort of trouble.

**Let's get our hands dirty with some web-scrapping then**

```yaml
soup= BeautifulSoup(page.content, 'html.parser')
soup
```
This should return all the html content of the page. In fact, you can go to the settings of the page, then Developer tools, then elements to verify that all of the html has indeed been copied from there. But why do we need the entire contents? All we need is some selective portion of the data and have it to ourselves in an understandable format.
As an example we will use today the web-page whose link was mentioned in the url. 

The website contains the full script of an episode of a world famous sitcom, Blackadder, starring Rowan Atkinson, who is more famous as Mr. Bean!!. Such a collection of dialogues can be very interesting for natural language generation purposes. Datasets similar in nature but of Harry Potter and Friends exist on Kaggle.

<img src="/assets/images/Web-scrapping.jpg">

Do you see the above picture? Look at the console and you will find that the immediate parent of the <p> is a div with a class ="entry-content lazyloaded". We want the text written in those <p> tags. We express such a wish in this format.
```yaml
script=soup.select('div.entry-content p')
script
```

We are pretty near there but not done yet. The readability can be made better. There are a lot of images in the script too. The get_text() function does a good job here.
```yaml
dialogues=[]
for i in script:
    dialogues.append(i.get_text())
    
dialogues
```

**And we assemble them together**

```yaml
file1 = open("Season 4.txt","a+")
for dialogue in dialogues:
  dialogue=dialogue.rstrip("\n")
  file1.write(dialogue)
  file1.write("\n")
file1.close()
```

So, that's just one episode done. We can do similarly for the others too. The full notebook is [here](https://github.com/SOUMEE2000/Natural-Language-Processing/blob/main/Created%20Datasets/Blackadder_webScraping.ipynb). 

The dataset I created is [here](https://www.kaggle.com/soumee2000/blackadderfullscriptsrowan-atkinson).
