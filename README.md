
# Bidirectional Sequence Models - Lab

## Introduction

In this lab, we'll learn to make use of **_Bidirectional Models_** to better classify sequences of text!

## Objectives

You will be able to:

* Understand and explain the basic architecture of a Bidirectional RNN. 
* Identify the types of problems Bidirectional approaches are best suited for. 
* Build and train Bidirectional RNN Models. 

## Getting Started

In this lab, we're going to use a **_Bidrectional LSTM Model_** to build a model that can identify toxic comments on social media. This dataset comes from the [Toxic Comment Classification Challenge on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) in partnership with Google and Jigsaw.

## The Problem

From the "Data" section of the Kaggle competition linked above:

"You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment."

This tells us a couple things about the problem, which will affect the overall architecture of our model. Although this may technically be a multiclass classification problem, in practice, our model will treat each comment as 6 concurrent instances of binary classification. This is because a toxic comment can fall into one or more of the categories listed above--for example, a  comment may be toxic, obscene, threatening, and insulting, all at the same time. 

Run the cell below to import everything we'll need for this lab. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
```

## Loading the Data

We'll start by loading in our training and testing data. You'll find the data stored inside of the file `data.zip` included in this repo. 

**_NOTE:_**  Before we can begin loading in the data using pandas, you'll first need to unzip the file. Go into the repo you've cloned and unzip the `data` folder into the same directory as this jupyter notebook now.

Next, we'll use pandas to load in our training and testing data, and then downsample to only 20% of the training data, in the interest of training time. 

In the cell below:

* Use pandas to read the training data from `data/train.csv`. Store this data in `train`.
* Set `train` equal to `train.sample(frac=0.2)`, so that we'll only use 20% of the data to train our model. Otherwise, training this model could take several hours. 


```python
train = None
train = None
```

Great! Next, we'll get the values for both our labels and the comments that will act as our training and testing data. We do this in order to get the data from pandas DataFrames to numpy arrays. 

In the cell below:

* Create an array called `list_classes` that contains the following classes, in this order:
    * `'toxic'`
    * `'severe_toxic'`
    * `'obscene'`
    * `'threat'`
    * `'insult'`
    * `'identity_hate'`
* Store the `.values` of the DataFrame that is returned by using `list_classes` to slice the label columns from `train` (slice `list_classes` from train, and then chain it with `.values`). Store this in `y`.
* Store the `.values` of `train['comment_text]` in `list_sentences_train`


```python
list_classes = None
y = None
list_sentences_train = None
```

According to the data dictionary for this Kaggle competition, there are no missing values in either the train or the test set. However, let's quickly double check, just to be sure!

Run the cell below to see if there are any missing values in either the training set.


```python
# Double check that there are no missing values in either training set
train['comment_text'].isna().any() 
```

## Preprocessing The Data

Next, We'll need to preprocess our data. We've already learned how to do most of this by working with NLTK--however, keras also contains some excellent preprocessing packages to help prepare text data.  Since we'll be feeding this data right into a model built with keras, this has the added benefit of ensuring that our data will be in a format that our model will be able to work with, meaning that we can avoid the weird bugs that sometimes occur when working with multiple different 3rd party libraries at the same time. 

Our preprocessing steps are:

1. **_Tokenize_** the data. 
2. Turn the tokenized text into **_Sequences_**
3. **_Pad_** the sequences so they're all the same length. 

In the cell below:

* Create a `Tokenizer`, which can be found inside the `text` module we imported at the top of the lab. Set the `num_words` parameter to `20000`, so that our model only uses the 20000 most common words. 
* Convert `list_sentences_train` to a python list, and then pass it in to our tokenizer's `.fit_on_texts()` method. 
* Call the tokenizer's `texts_to_sequences()` method on `list_sentences_train` and store the result returned in `list_tokenized_train`
* Use the `sequence` module's `pad_sequences()` method and pass in `list_tokenized_train`, as well as the parameter `maxlen=100`. Store the result returned in `X_t`.


```python
# NOTE: This cell may take a little while to run!
tokenizer = None

list_tokenized_train = None
X_t = None
```

## Creating Our Model

Now that we've loaded and preprocessed our data, we're ready to begin designing our model. By now, working with keras to create and compile a model will probably feel familiar to you. To keep things simple, we've left the name of each layer below. Your job will be to create each layer, and specify the previous layer that acts as it's input (which is why so many of the layers are called `x` below--you've probably noticed this simplifies the creation process by eliminating the need to keep track of which layer is which at any given point). 

In the cell below:

* Set the `embedding_size` to `128`
* Create an `Input` layer that takes in data of `shape=(100,)`
* Next, create an `Embedding` layer and pass in `20000` and `embedding_size` as parameters. Make sure to specify that the Embedding layer takes in the output of the input layer as its input by ending the line with `(input_)`
* Create a `Bidirectional` layer. Inside this layer, pass in an `LSTM()`. The parameters for the LSTM should be `25`, and `return_sequences=True`. 
* Create a `GlobalMaxPool1D` Layer
* Create a `Dropout` layer, and pass in `0.5` as a parameter. 
* Create a `Dense` layer with `50` neurons, and set the `activation` to `'relu'`
* Create another `Dropout` layer, and pass in `0.5` as the parameter. 
* Create a `Dense` layer with `6` neurons, and set the `activation` to `'sigmoid'`. 
* Create a `Model` and set the `inputs` to `input_` and `outputs` to `x`.


```python
embedding_size = None
input_ = None
x = None
x = None
x = None
x = None
x = None
x = None
x = None

model = None
```

Great! Now that we've created our model, we still need to compile it.  

In the cell below:

* Call `model.compile` and pass in the following parameters:
    * `loss='binary_crossentropy'`
    * `optimizer='adam'`
    * `metrics=['accuracy']`

Now, let's take a look at the model we've created. In the cell below, call `model.summary()`.

## Setting Some Checkpoints

Training models like this can be tricky. Because of that, we'll make use of **_Checkpoints_** to help us periodically save our work in case things go wrong during training. 

We'll create two different types of checkpoints below:

* A `ModelCheckpoint` that saves the best weights for our model at any given time inside an `hdf5` file. This way, if our model's performance starts to degrade at any point, we can always reload the weights from a snapshot of when it had the best possible performance. 

* An `EarlyStopping` checkpoint, which will stop the training early if the model goes for a certain number of epochs without any progress. 

For this lab, we'll only be training the model for a single epoch, so we don't actually need to use these checkpoints. However, on the job, models like this are often trained for days at a time. With training times that long, checkpoints are absolutely crucial to avoid losing days of work. There are few things more frustrating than seeing that you model was performing really well 2 days ago, but has since began to have performance degrade due to overfitting, and you have to start the training over because you forgot to set checkpoints!

Run the cells below to create the checkpoints and store them in an array that we'll pass in during training. For more information on the checkpoints we've created, see the [Keras callbacks documentation](https://keras.io/callbacks/#earlystopping).


```python
checkpoints_path = 'weights_base.best.hdf5'
checkpoint = ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
```


```python
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=25)
```


```python
callbacks = [checkpoint, early_stopping]
```

## Training the Model

Now, we're ready to train our data. Because our model contains over 1.9 million trainable parameters, this will take a little while to train! 

In the cell below:

* Call `model.fit()` and pass in the following parameters:
    * `X_t`
    * `y`
    * `batch_size=32`
    * `epochs=1`
    * `validation_split=0.1`
    * `callbacks=callbacks`
    
**_NOTE:_** Running the cell below may take 15+ minutes, depending on your machine. Run it, then go get a coffee!

Validation accuracy of over 97.8% when trained on only 20% of the data--this is excellent! If you train on the entire training set, you'll see that we achieve over 98% accuracy after only 1 epoch of training. It's safe to say our model works pretty well!

## Summary

In this lab, we incorporated everything we've learned about sequence models and embedding layers to build a Bidirectional LSTM Network to successfully classify toxic comments from wikipedia!
