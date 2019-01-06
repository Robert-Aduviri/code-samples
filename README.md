# Code samples

These are some code samples from various research projects, side projects and competitions. 

## Deep Learning

### [Variational Sparse Coding](https://github.com/Alfo5123/Variational-Sparse-Coding/)
Implementation of the "Variational Sparse Coding" [paper](https://openreview.net/pdf?id=SkeJ6iR9Km) as part of the ICLR 2019 Reproducibility Challenge. [[code](https://github.com/Alfo5123/Variational-Sparse-Coding/tree/master/src)]

<p align="center">
 <img src="https://github.com/Alfo5123/Variational-Sparse-Coding/blob/master/results/images/latent8alpha001.gif" width="50%">
</p>

### [Neural Machine Translation](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/NMT)
Sequence-to-sequence recurrent neural network (bidirectional LSTM) with Global Attention ([Luong et al., 2015](https://arxiv.org/abs/1508.04025)) and Beam Search implemented in PyTorch. ~41 BLEU in 110K-sentences English-Spanish [corpus](http://www.manythings.org/anki/).

<p align="center"><img src="https://raw.githubusercontent.com/Robert-Alonso/code-samples/master/Deep%20Learning/NMT/attention-visualization-sample.png" width="50%"></p>
  
### [Pokédex: Transfer Learning for Image Classification](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Pok%C3%A9dex%20CNN)
Pokémon image classification with transfer learning from ImageNet-pretrained MobileNet convolutional neural network ([Howard et al., 2017](https://arxiv.org/abs/1704.04861)). ~82% accuracy with 27 classes and 3.8K web-scraped images. Deployed in Flask with React JS for the SPA user interface. Presented at [Infosoft 2017](http://convencion.pucp.edu.pe/infosoft/ediciones-anteriores/infosoft-2017-edicion-centenario/?seccion=programa&fecha=2017-09-06) and [Hack Faire 2017](https://www.facebook.com/HackSpacePeru/posts/1785114531498727). [[slides](https://github.com/Robert-Alonso/code-samples/blob/master/Deep%20Learning/Pok%C3%A9dex%20CNN/How%20to%20make%20a%20Pok%C3%A9dex.pdf)] [[demo](https://robert-alonso.github.io/React-Pokedex/)]

<p align="center"><img src="https://raw.githubusercontent.com/Robert-Alonso/code-samples/master/Deep%20Learning/Pok%C3%A9dex%20CNN/image-classifier-sample.png" width="50%"></p>

### [Job2Vec: Job matching from word-embeddings](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Semantic%20Text%20Matching)
Information retrieval system between job descriptions and applicant profiles textual description matching based on Word2Vec and Doc2Vec ([Le & Mikolov, 2014](https://arxiv.org/abs/1405.4053)) semantic search and string matching algorithms for out-of-vocabulary misspelled words, constructed over an inverted index for efficient look-up. Presented at [WAIMLAp 2017](http://grpiaa.inf.pucp.edu.pe/waimlap2017/?page_id=211) and [Hack Faire 2017](https://www.facebook.com/HackSpacePeru/posts/1785114531498727). [[poster](https://github.com/Robert-Alonso/code-samples/blob/master/Deep%20Learning/Semantic%20Text%20Matching/Job%20Matching%20Poster.pdf)]

<p align="center"><img src="https://raw.githubusercontent.com/Robert-Alonso/code-samples/master/Deep%20Learning/Semantic%20Text%20Matching/word-embedding-space.png"></p>

### [Large-scale Text Classification](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Text%20Classification)
Commodity description classification using recurrent neural networks (bidirectional LSTM) implemented in PyTorch with FastText pretrained word embeddings ([Joulin et al., 2016](https://arxiv.org/abs/1607.01759)). ~92% top-5 accuracy with 3762 classes and 30.6M text descriptions.

<p align="center"><img src="https://raw.githubusercontent.com/Robert-Alonso/code-samples/master/Deep%20Learning/Text%20Classification/text-classifier-sample.png" width="80%"></p>

### [Genomics time-series pattern-recognition with Convolutional Neural Networks](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Genomics)
Convolutional neural networks architecture experimentation for genomic sequence pair binary classification with high imbalance (0.07% positive classes). ~78.5 F1-Score for ~200k pairs of sequences.
  
<p align="center"><img src="https://raw.githubusercontent.com/Robert-Alonso/code-samples/master/Deep%20Learning/Genomics/classifier-sample.png" width="40%"></p>
  
### [Fully-connected Autoencoder for MNIST](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/MNIST%20Autoencoder)
Fully-connected autoencoder for MNIST dataset with a bottleneck of size 20 implemented in PyTorch, based on DeepBayes 2018 [practical assignment](https://bayesgroup.github.io/deepbayes-school/2018/task/description/). 0.00069 L2 reconstruction loss + L1 regularization loss. t-SNE dimensionality reduction for bottleneck features visualization.

<p align="center"><img src="https://raw.githubusercontent.com/Robert-Alonso/code-samples/master/Deep%20Learning/MNIST%20Autoencoder/autoencoder-visualization.png" width="70%"></p>

## Data Science Competitions

### [Movistar Data Talent 2018](https://github.com/Robert-Alonso/Movistar-Datathon/blob/master/src/utils_r.py)
Main preprocessing and main cross validation loop with LightGBM

### [Data Science Game Finals 2018](https://github.com/Robert-Alonso/DSG-2018-Finals/blob/master/src/utils_r.py)
LSTM conditioned on a structured embedding network implemented in PyTorch (refactored version)

### [Data Science Game Qualifiers 2018](https://github.com/Robert-Alonso/DSG-2018-Qualifiers/blob/master/src/multimodal.py)
LSTM conditioned on a structured embedding network implemented in PyTorch


  
### Other competitions
  - [BBVA Challenge 2017](https://github.com/Robert-Alonso/code-samples/tree/master/Data%20Science%20Competitions/BBVA%20Challenge)
  - [DrivenData Competition](https://github.com/Robert-Alonso/code-samples/blob/master/Data%20Science%20Competitions/DrivenData%20Competition.ipynb)
  - [Interbank Datathon](https://github.com/Robert-Alonso/code-samples/blob/master/Data%20Science%20Competitions/Interbank%20Datathon.ipynb)
  - [Kaggle Bulldozers Competition](https://github.com/Robert-Alonso/code-samples/blob/master/Data%20Science%20Competitions/Kaggle%20Bulldozers.ipynb)
  - [Kaggle Homesite Insurance Competition](https://github.com/Robert-Alonso/code-samples/blob/master/Data%20Science%20Competitions/Kaggle%20Homesite%20Insurance.ipynb)
  - [Kaggle NYC Taxi Trip Duration Competition](https://github.com/Robert-Alonso/code-samples/blob/master/Data%20Science%20Competitions/Kaggle%20NYC%20Taxi%20Trip%20Duration.ipynb)

## Coursework
  - [CIFAR-10 with CNNs](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Coursework/CIFAR-10%20CNN) (Tensorflow)
  - [CelebA Image Generation with DCGANs](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Coursework/DCGAN) (Tensorflow)
  - [Dogs vs Cats with CNNs + Transfer Learning](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Coursework/Keras%20CNN) (Keras)
  - [English-French Machine Translation with Seq2seq RNNs](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Coursework/LSTM%20Machine%20Translation) (Tensorflow)
  - [The Simpons TV Scripts Text Generation with LSTM RNNs](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Coursework/LSTM%20Text%20Generation) (Tensorflow)
  - [LibriSpeech Speech Recognition with GRU RNNs and CNNs](https://github.com/Robert-Alonso/code-samples/tree/master/Deep%20Learning/Coursework/CNN%20GRU%20Speech%20Recognition) (Keras)

## Miscellaneous
  - [Music automatic matching with audio descriptors](https://github.com/Robert-Alonso/code-samples/tree/master/Miscellaneous/Audio%20matching)
  - [Nearest Neighbor Collaborative Filtering](https://github.com/Robert-Alonso/code-samples/blob/master/Miscellaneous/Nearest-neighbor%20Collaborative%20Filtering.ipynb)
  - [Geospatial Visualization](https://github.com/Robert-Alonso/code-samples/blob/master/Miscellaneous/Geospatial%20Visualization.ipynb)

