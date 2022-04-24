# CDA_CT3
Files for third assignment of CDA

## Word to Vec Alogorithm 
Word to Vec is a technique for natural language processing published in 2013. The Word to Vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, Word to Vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors. Word to Vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word to Vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.
Word to Vec is not a singular algorithm, rather, it is a family of model architectures and optimizations that can be used to learn word embeddings from large datasets. Embeddings learned through Word to Vec have proven to be successful on a variety of downstream natural language processing tasks.

## Count Vectorizer 
Machines cannot understand characters and words. So when dealing with text data we need to represent it in numbers to be understood by the machine. Countvectorizer is a method to convert text to numerical data. Countvectorizer makes it easy for text data to be used directly in machine learning and deep learning models such as text classification.
The CountVectorizer is the simplest way of converting text to vector. It tokenizes the documents to build a vocabulary of the words present in the corpus and counts how often each word from the vocabulary is present in each and every document in the corpus. Thus, every document is represented by a vector whose size equals the vocabulary size and entries in the vector for a particular document show the count for words in that document. When the document vectors are arranged as rows, the resulting matrix is called document-term matrix; it is a convenient way of representing a small corpus.

### Code-
The code below shows how to use Count Vectorizer in Python.
```
#list of text documents
text = ["John is a good boy. John watches basketball"]

vectorizer = CountVectorizer()
#tokenize and build vocab
vectorizer.fit(text)

print(vectorizer.vocabulary_)

#encode document
vector = vectorizer.transform(text)
#summarize encoded vector
print(vector.shape)
print(vector.toarray())
```

## TFID Vectorizer 
TF-IDF is a method which gives us a numerical weightage of words which reflects how important the particular word is to a document in a corpus. A corpus is a collection of documents. Tf is Term frequency, and IDF is Inverse document frequency. This method is often used for information retrieval and text mining.
Term frequency-inverse document frequency is a text vectorizer that transforms the text into a usable vector. It combines 2 concepts, Term Frequency (TF) and Document Frequency (DF).
The term frequency is the number of occurrences of a specific term in a document. Term frequency indicates how important a specific term in a document. Term frequency represents every text from the data as a matrix whose rows are the number of documents and columns are the number of distinct terms throughout all documents.
Document frequency is the number of documents containing a specific term. Document frequency indicates how common the term is.
Inverse document frequency (IDF) is the weight of a term, it aims to reduce the weight of a term if the term’s occurrences are scattered throughout all the documents.
In order to process natural language, the text must be represented as a numerical feature. The process of transforming text into a numerical feature is called text vectorization. TF-IDF is one of the most popular text vectorizers, the calculation is very simple and easy to understand. It gives the rare term high weight and gives the common term low weight

### Code
Below is an example which depict how to compute tf-idf values of words from a corpus using tfid vectorizer: 
```
# import required module
from sklearn.feature_extraction.text import TfidfVectorizer
  
# assign documents
d0 = 'Geeks for geeks'
d1 = 'Geeks'
d2 = 'r2j'
  
# merge documents into a single corpus
string = [d0, d1, d2]
  
# create object
tfidf = TfidfVectorizer()
  
# get tf-df values
result = tfidf.fit_transform(string)
  
# get idf values
print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names(), tfidf.idf_):
    print(ele1, ':', ele2)
  
# get indexing
print('\nWord indexes:')
print(tfidf.vocabulary_)
  
# display tf-idf values
print('\ntf-idf value:')
print(result)
  
# in matrix form
print('\ntf-idf values in matrix form:')
print(result.toarray())
```

## Regularization 
In regression analysis, the features are estimated using coefficients while modelling. Also, if the estimates can be restricted, or shrinked or regularized towards zero, then the impact of insignificant features might be reduced and would prevent models from high variance with a stable fit. 
Regularization is the most used technique to penalize complex models in machine learning, it is deployed for reducing overfitting (or, contracting generalization errors) by putting network weights small. Also, it enhances the performance of models for new inputs.
In simple words, it avoids overfitting by panelizing the regression coefficients of high value. More specifically, It decreases the parameters and shrinks (simplifies) the model. This more streamlined model will aptly perform more efficiently while making predictions.
Since, it makes the magnitude to weighted values low in a model, regularization technique is also referred to as weight decay.
Moreover, Regularization appends penalties to more complex models and arranges potential models from slightest overfit to greatest. Regularization assumes that least weights may produce simpler models and hence assist in avoiding overfitting.
The model with the least overfitting score is accounted as the preferred choice for prediction. 
In general, regularization is adopted universally as simple data models generalize better and are less prone to overfitting.

## L1 and L2 Regularization 
A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression.
The key difference between these two is the penalty term.
Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.

### L1 Regularization
L1 regularization is the preferred choice when having a high number of features as it provides sparse solutions. Even, we obtain the computational advantage because features with zero coefficients can be avoided.

### L2 Regularization
L2 regularization can deal with the multicollinearity (independent variables are highly correlated) problems through constricting the coefficient and by keeping all the variables. 
L2 regression can be used to estimate the significance of predictors and based on that it can penalize the insignificant predictors.
A regression model that uses L2 regularization techniques is called Ridge Regression.
