# -*- coding: utf-8 -*-
"""
***********************************************************************************
Created on Sun Oct  2 08:10:35 2022

@author: Aaron Ogle

PROTEIN FOLD CLASSIFICATION USING DEEP NEURAL NETWORKS

In this program we will be classifying protein folds based off the protein's
amino acid sequence.  The data set that we will be using is SCOPe 1.55.  First,
we need to parse, clean, and re structure the data so that it can be input
into our different deep learning models.

We will perform some exploratory data analysis during the cleaning process
which will provide new insight into the structure of our data.  Through the
cleaning process we decided that instead of classifying proteins into 138
different folds it would be much more feasible to find the "top 10" fold
classes that contained the most samples so that our models would be able
to learn enough about the samples so that it could classify them more
accurately.  With this information we found the top 10 folds and removed
the other folds from the data set that we will be providing the models.
After we cleaned, explored, and re structured the data we then input the 
data into our deep learning models for training and testing.

The architectures of the deep learning models are based off those that were
found in related works.  The specific related works are mentioned in the
comments before the actual implementation of the model.  To create the 
models we rely on the functionality provided by the keras deep learning 
library.

After training and testing our models we will generate plots that highlight
the values for loss and accuracy through each of the epochs the algorithms
train/test through.  We will also display the final testing accuracy of the model
so that we can compare these results among the different models and discuss
the results in our paper. 

***********************************************************************************
"""
# import matplotlib.pyplot so that we can use it for plotting
import matplotlib.pyplot as plt
# import statistics so we can find the median value in a list
import statistics
# Find the number of samples that we have of each fold
from collections import Counter
# Since we are working with text data we will need the below to tokenize the chars
from keras.preprocessing.text import Tokenizer
# Our text data varies in its length so we will use the 
# below for padding
from keras_preprocessing.sequence import pad_sequences
# Now we need to import the model and layers 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Embedding, Conv1D, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
# Needed for numpy arrays
import numpy as np


"""
#########################################################################
Function: parse_data()

Parameters: Filename

Returns: Parsed data

The function get_data will accept a filename as input and will
then parse out the relevant data that we need and place that
data into specific lists.
#########################################################################
"""

def parse_data(file):
    # Open the file and read the content of it
    
    with open(file, 'r') as f:
        data_set = f.read()
        
    # For our dataset, the entries we are interested in
    # are all separated by the '>' character so we want
    # to split the data by that character
    
    data_set = data_set.split('>')
    
    # Now we want to parse out the different amino acid
    # sequences and place those values into their own list
    # Note that our sequences are after a '}' character so
    # we are going to partition our rows by that value
    
    seq = []
    for row in data_set:
        seq.append(row.partition("}")[2])
        
    # Next we need to parse out the different fold groups
    # Note that the folds are between the values 'a.' and '('
    # so we want to return the values that lie between those
    # two chars
    
    folds = []
    for row in data_set:
        folds.append(row[row.find('a.')+2 : row.find('(')])
        
    # Next we will parse out the different protein families
    # in case we need to use those later on
    # Similar logic as in the above except the families
    # are between the ')' and '{' characters
    
    families = []
    for row in data_set:
        families.append(row[row.find(')')+2 : row.find('{')])
        
    # Return the parsed out values for our sequences, our folds
    # and our families
    
    return seq, folds, families

"""
#########################################################################
Function: EDA(seq, fold, fam)

Parameters: The sequence, fold, and family datasets that were generated
by our parse_data function

Returns: cleaner versions of our data and will also print out
statistical / exploratory information about our data

The function EDA will allow us to explore our dataset more and will help
us answer some foundational questions that we are asking ourselves such
as

1) What is the maximum sequence length in our dataset?
2) What is the minimum sequence length in our dataset?
3) What is the average sequence length in our dataset?
    
These are the main questions that will help us since it will be very
useful information to have once we are trying to send the sequence data
through our DNN models.
#########################################################################
"""

def EDA(seq, fold, fam):
    
    # Note that I will leave the majority of this commented out
    # so that the noise it generates in the program is reduced
    # I wanted to leave this here in order to show the data
    # checks and further cleansing of our dataset
    
    """
    First we will check the sequences to find our
    minimum and maximum length
    
    
    seq_length = [len(row) for row in seq]
    print("Minimum sequence length is: ", min(seq_length))
    print("Maximum sequence length is: ", max(seq_length))
    
    
    From the above we can see we have sequences of length 0 
    we will want to remove these values from our data
    
    the below for loop was used to find the indices where
    we have length 0, I will leave this commented out as
    it is not useful information outside of this.
    
    for i in range(0,len(seq)):
        if len(seq[i]) == 0:
            print(i)
    """
    
    # From the above we found issues with indexes 702,
    # 701, and 0 so we are removing those from our datasets
    seq.pop(702)
    seq.pop(701)
    seq.pop(0)
    fold.pop(702)
    fold.pop(701)
    fold.pop(0)
    fam.pop(702)
    fam.pop(701)
    fam.pop(0)
    
    """
    Check our cleaned sequences and see what the min, max and mean
    of the sequence lengths are
    
    #seq_length = [len(row) for row in seq]
    #print(min(seq_length), max(seq_length), statistics.mean(seq_length))
    
    From the above we found:
        
    MAX Sequence Length = 904
    MIN Sequence Length = 29
    AVG Sequence Length = 151
    
    After further tests, we really do not need the information after 
    the '.' in our original fold class list.
    
    In the below, I am dropping everything to the right of the '.'
    character in the fold classification as all we really need is the actual
    fold class.
    """
    
    fold_class = []
    for row in fold: 
        fold_class.append((row.split('.'))[0])
        
    """
    For some reason, during our data cleaning process
    {Thermus aquaticus}
    lrpdqwadyraltgdesdnlpgvkgigektarklleewgsleallknldrlkpairekil
    ahmddlklswdlakvrtdlplevdfakrrepdrerlraflerlefgsllhefglle 
    
    became the value at index 700 instead of the actual class so
    we are going to remove this value.
    """
    
    fold_class.pop(700)
    seq.pop(700)
    
    int_fold_class = []
    for row in fold_class:
        int_fold_class.append(int(row))
        
    fold = int_fold_class
    
    # Plot the fold data so that we can show how unproportional
    # the dataset is (there are multiple classes that only have
    # a few training example)
    
    # We use 138 bins since we have 138 fold classes
    
    """
    plt.hist(fold, bins=138)
    plt.xlabel('Fold Number') 
    plt.ylabel('Number of Samples') 
    plt.title("Number of Samples by Fold")
    plt.show()
    """
    
    # Since our data is heavily skewed and has certain folds
    # containing far more samples than others, we are going
    # to focus on just classifying a group of 10 different
    # kinds of folds.  We will select the top 10 folds that
    # have the most samples for us to use
    
    """
    From the above we see our top ten are:
        
    4: 122, 
    1: 75, 
    39: 66, 
    3: 47, 
    118: 43, 
    26: 34, 
    45: 32, 
    24: 31, 
    2: 28, 
    60: 24
    """
    
    # Create a list that houses the fold number for
    # our top 10 folds
    top_10 = [4,1,39,3,118,26,45,24,2,60]
    
    # Create a list of indices that will need to be
    # dropped from our seq, fold, and fam lists
    drop = []
    
    # Find the indices of the folds that are not in our top ten group
    for i in range(len(fold)):
        if fold[i] not in top_10:
            drop.append(i)
    
    # We need to delete these in reverse order so that we do not
    # delete values that we actually need
    for index in sorted(drop, reverse=True):
        del seq[index]
        del fold[index]
        del fam[index]
        
    """
    Relabel Folds
    
    Here we are going to change the labels of the folds since we had
    to cut some of the folds out that we are going to classify
    
    New fold labeling
    1 -> 1
    2 -> 2
    3 -> 3
    4 -> 4
    24 -> 5
    26 -> 6
    39 -> 7
    45 -> 8
    60 -> 9
    118 -> 10
    """
    for i in range(len(fold)):
        if fold[i] == 24:
            fold[i] = 5
        elif fold[i] == 26:
            fold[i] = 6
        elif fold[i] == 39:
            fold[i] = 7
        elif fold[i] == 45:
            fold[i] = 8
        elif fold[i] == 60:
            fold[i] = 9
        elif fold[i] == 118:
            fold[i] = 10
    
    """
    Test for various attributes now that we have deleted data
    length = []
    for row in seq: length.append(len(row))
    print(max(length), min(length), statistics.mean(length), len(fold))
    
    With data being removed we are only left with 502 total samples
    
    The max length of a sequence is 904 characters 
    The min length of a sequence is 31 characters
    The average length of a seqence is 131 characters
    """
    """
    # Plot the data after dropping all but the top ten results
    plt.hist(fold, bins=10)
    plt.xlabel('Fold Number') 
    plt.ylabel('Number of Samples') 
    plt.title("Number of Samples by Fold")
    plt.show()
    """
    
    return seq, fold, fam

def data_transformation(x, y):
    # From our exploratory data analysis we found that the
    # average sequence length in our dataset is ~130
    # This was originally going to be the max length of a sequence
    # but through experimentation it appears that if we cutoff
    # a sequence at 200 characters then we get higher accuracy results
    max_length = 200
    
    # We are passing our text data in as a list and we need
    # to have each character broken out of that so we
    # create a list and then append it with the sequence
    # as a list itself -> this creates a list of lists
    # (matrix) so that we can pass it to the tokenizer
    x_char_list = []
    for row in x:
        x_char_list.append(list(row))
    
    # convert our sequence text into integer values
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(x_char_list)
    sequences = tokenizer.texts_to_sequences(x_char_list)
    
    # Now we need to pad the data with zeros depending on
    # our max length
    data_set = pad_sequences(sequences, max_length)
    max_word_index = len(tokenizer.word_index)
    #print(max_word_index) -> 22 (number of distinct characters stored we need this for our input)
    
    # Make the labels our y values a numpy array
    labels = np.asarray(y)
    
    """
    Check the sizes of our tensors
    print('Shape of data tensor: ', data_set.shape)
    print('Shape of label tensor: ', labels.shape)
    """
    
    # We need to split up the data for training and testing
    # but our data is ordered so we need to randomly partition
    # the data and move it around so that it is in a random order
    
    indices = np.arange(data_set.shape[0])
    np.random.shuffle(indices)
    data_set = data_set[indices]
    labels = labels[indices]
    
    x_train = data_set[:400]
    y_train = labels[:400]
    x_val = data_set[400:]
    y_val = labels[400:]
    
    return x_train, y_train, x_val, y_val

def cnn(max_length, x_train, y_train, x_test, y_test):
    
    
    """
    #########################################################################################
    CONVOLUTIONAL NEURAL NETWORK
    
    Note that the original architecture of this network was based off the work done in 
    the following work:
        
    DeepSF: deep convolutional neural network for mapping protein sequences to folds
    
    However, through trial and error we found a different architecture that performed better.
    #########################################################################################
    """
    
    
    model = Sequential()
    model.add(Embedding(23, 10, input_length=max_length))
    model.add(Conv1D(filters=128, kernel_size=20, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(11, input_dim=max_length,activation='softmax'))
    
    model.compile(optimizer='rmsprop',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    early_stopping = EarlyStopping(monitor='acc', mode='max',
                                                   restore_best_weights=True, patience=5)
    
    hist = model.fit(x_train, 
                     y_train, 
                     validation_data = (x_test, y_test), 
                     epochs=50, 
                     callbacks = early_stopping)
    
    results = model.evaluate(x_test, y_test)
    
    print(f"Model Accuracy: {round(results[1]*100,2)}%")
    
    # Define variables to track accuracy and loss so that these can be plotted
    train_acc = hist.history['acc']
    train_loss = hist.history['loss']
    test_acc = hist.history['val_acc']
    test_loss = hist.history['val_loss']
    
    # Plot Accuracy
    plt.plot(train_acc, label = "Training Accuracy")
    plt.plot(test_acc, label = "Test Accuracy")
    plt.title("1D CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    # Plot Loss
    plt.plot(train_loss, label = "Training Loss")
    plt.plot(test_loss, label = "Test Loss")
    plt.title("1D CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
def bilstm(max_length, x_train, y_train, x_test, y_test):
    
    
    """
    #########################################################################################
    BIDIRECTIONAL LONG SHORT-TERM MEMORY RECURRENT NEURAL NETWORK
    
    Note that the architecture of this network is based off the work done in the following:
        
    A Novel Approach to Protein Folding Prediction based on Long Short-Term Memory Networks: 
    A Preliminary Investigation and Analysis
    
    Instead of using just a LSTM model we instead go for the Bidirectional LSTM model; the
    reasoning for this is due the point made in our textbook "Deep Learning" by Goodfellow
    et al that sometimes when using sequence data the output we are trying to predict is
    dependent on the whole input sequence and not just a portion/multiple portions of this.  
    
    With that we instead use the Bidirectional based model that is discussed in the textbook
    along with the LSTM method.
    #########################################################################################
    """
    
    model = Sequential()
    model.add(Embedding(23, 10, input_length=max_length))
    model.add(Dense(100, activation='relu'))
    model.add(Bidirectional(LSTM(10, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(11, input_dim=max_length,activation='softmax'))
    
    model.compile(optimizer='RMSprop',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    early_stopping = EarlyStopping(monitor='acc', mode='max',
                                                   restore_best_weights=True, patience=5)
    

    hist = model.fit(x_train, 
                     y_train, 
                     validation_data = (x_test, y_test), 
                     epochs=50, 
                     callbacks = early_stopping)
    
    results = model.evaluate(x_test, y_test)
    
    print(f"Model Accuracy: {round(results[1]*100,2)}%")
    
    # Define variables to track accuracy and loss so that these can be plotted
    train_acc = hist.history['acc']
    train_loss = hist.history['loss']
    test_acc = hist.history['val_acc']
    test_loss = hist.history['val_loss']
    
    # Plot Accuracy
    plt.plot(train_acc, label = "Training Accuracy")
    plt.plot(test_acc, label = "Test Accuracy")
    plt.title("Bidirectional LSTM RNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    # Plot Loss
    plt.plot(train_loss, label = "Training Loss")
    plt.plot(test_loss, label = "Test Loss")
    plt.title("Bidirectional LSTM RNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return results

def cnnrnn(max_length, x_train, y_train, x_test, y_test):
    
    
    """
    #########################################################################################
    COMBINED 1D CNN AND BIDIRECTIONAL LSTM RNN ARCHITECTURE
    #########################################################################################
    """
    
    model = Sequential()
    model.add(Embedding(23, 10, input_length=max_length))
    model.add(Conv1D(filters=128, kernel_size=20, activation='relu'))
    model.add(Bidirectional(LSTM(10, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(11, input_dim=max_length,activation='softmax'))
    
    model.compile(optimizer='RMSprop',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    early_stopping = EarlyStopping(monitor='acc', mode='max',
                                                   restore_best_weights=True, patience=5)
    
    
    hist = model.fit(x_train, 
                     y_train, 
                     validation_data = (x_test, y_test), 
                     epochs=50, 
                     callbacks = early_stopping)
    
    results = model.evaluate(x_test, y_test)
    
    print(f"Model Accuracy: {round(results[1]*100,2)}%")
    
    # Define variables to track accuracy and loss so that these can be plotted
    train_acc = hist.history['acc']
    train_loss = hist.history['loss']
    test_acc = hist.history['val_acc']
    test_loss = hist.history['val_loss']
    
    # Plot Accuracy
    plt.plot(train_acc, label = "Training Accuracy")
    plt.plot(test_acc, label = "Test Accuracy")
    plt.title("1D CNN and Bidirectional LSTM RNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    # Plot Loss
    plt.plot(train_loss, label = "Training Loss")
    plt.plot(test_loss, label = "Test Loss")
    plt.title("1D CNN and Bidirectional LSTM RNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return results
    
    
def main():
    
    """
    ########################################################################
    Data Parsing
    ########################################################################
    
    We use the parse_data function in order to split our data
    into the relevant parts that we need.  We parse out the
    sequence values, the fold classifications, and then the
    different protein families.  
    """
    protein_data = 'protein_data.txt'
    seq, fold, fam = parse_data(protein_data) 
    

    """
    ########################################################################
    Exploratory Data Analysis
    ########################################################################
    
    In the below section we will us a function to explore 
    what our data looks like and some of the statistical features
    such as min, max, average, etc that lies in our
    data.  This can help us with the data cleaning
    process so that we can remove outliers in our
    dataset.
    
    Please see above function EDA (exploratory data analysis)
    for further documentation.
    """
    
    seq, fold, fam = EDA(seq, fold, fam)
    
    """
    ########################################################################
    Data Transformation
    ########################################################################
    
    Now that the data has been cleaned and restructured we need to transform
    it into a set and consistent structure that we can send to each one
    of our deep learning models.  The data_transformation function will
    tokenize our characters into numerical values, perform zero padding on 
    sequences that are less than 200 characters long, and will finally split 
    the data into training and testing sets that can be fed to our deep 
    learning models.
    """
    
    x_train, y_train, x_test, y_test  = data_transformation(seq, fold)
    
    """
    ########################################################################
    DNN Models
    ########################################################################
    
    In the below section we can send our training and testing data to the
    different models.
    
    The first model is a 1D CNN
    
    The second model is a Bidirectional LSTM RNN
    
    The third model is a combined 1D CNN Bidirectional LSTM RNN model
    
    For additional information on the design of the architectures please
    see the comments section that is above each of the different models.
    """
    # Set a max length for our sequence data - value was found through
    # experimenting with different values
    max_length = 200
    print(cnn(max_length, x_train, y_train, x_test, y_test))
    #print(bilstm(max_length, x_train, y_train, x_test, y_test))
    #print(cnnrnn(max_length, x_train, y_train, x_test, y_test))
    

        
main()
    
    
    


