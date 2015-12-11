# Semantic Role Labeling using DeepLearning
This is a deep learning architecture system which can do:
  - Language Modeling
  - Semantic Role Labeling
 
The System is the part of implementation of the publication ["A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning"](http://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf) by [Collobert, et.al.](http://ronan.collobert.com/)


### EXTENSION:
On top of the implementation we use the system to train on a input domain with lot of training instance and adapt the trained model to a new domain with less number of annotated sentences. 

#### STEP 1: 
    Train the Semantic Role Labeling Model on the input domain.

#### STEP 2:
    The idea is to retrain the word representations of target domain words by the Language Model. (i.e Lets say the input domain is a list of new articles on General Domain, Here the word representation of President Obama will be close to Politicians but our target domain is a law domain, now the word representation of President Obama will be tuned to appear close to other Lawyers).

#### STEP 3:
    Use the Trained Semantic Role Labeling Network in STEP1 and tuned word vectors in STEP 2 to retrain the model on the target domain.


### USAGE:

1. Language Model : th LanguageModel.lua
2. SRL on input domain : th SRL.lua "train/test" sentence_start sentence_end [clean]
3. SRL on target domain : th DomainAdaptor.lua "train/test" sentence_start sentence_end [clean]

### CONTRIBUTORS:

[Santhosh Kumar Manavasi Lakshminarayanan] (https://github.com/santhoshkumarml)

[Manoj Alwani](https://github.com/manojstonybrook)

[Sriganesh Navaneethakrishnan](https://github.com/SriganeshNk)
