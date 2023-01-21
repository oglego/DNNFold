#######################################################################################

PROTEIN FOLD CLASSIFICATION USING DEEP NEURAL NETWORKS

#######################################################################################

DESCRIPTION:

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

#######################################################################################

INSTALLING / RUNNING 

To run this program, you will need to have the following python libraries installed:

1) keras
2) matplotlib
3) statistics
4) collections
5) numpy

The program will run as is without statistics/collections, but if the user wants to
use the "exploratory data analysis" function then those libraries will need to be added.

For this work, we used the alpha proteins from the SCOPe 1.55 dataset, this data is included
in the protein_data.txt file.  For the program to run, this txt file needs to be in the 
same file path/directory as the .py file.

The program currently is set to run the 1D CNN model, however, by simply commenting
in and out the other model functions the user can decide on what other model to use.

#######################################################################################






