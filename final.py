from nltk import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import numpy as np

from nltk.stem import WordNetLemmatizer

#%%
#gensim library
def preprocess(docx):    
    lemmatizer=WordNetLemmatizer()
    docx=docx.lower()
    docx=re.sub('[^a-zA-Z .]',' ',docx)
    all_words=word_tokenize(docx)
    root_words=[lemmatizer.lemmatize(j) for j in all_words]
    string=' '.join(root_words)
    
    return(string)
    
#%%
#n-gram(Bigram) model
def ngram(corpus):
    new_vocab=word_tokenize(corpus)
    fd=FreqDist(new_vocab)
    words = [w for w in fd.keys()]
    most=[]
    for wr in words:
        most.append([wr,fd[wr]])
    most=sorted(most,key=lambda i:i[1],reverse=True)
    most=[a for a in most if a[0] not in set(stopwords.words('english')) ]
    most=most[:10]
  
    x=[]
    for b,c in most:
        x.append(b)
    
    new=[]
    for i in x:
        new.append([i,[n for n,val in enumerate(new_vocab) if val==i]])
    y=[]
    for j,k in new:
        y.append(k)

#bi-gram model:    
    master_prev=[]
    for z in y:
        for p in z:
            master_prev.append([(new_vocab[p-1]),(new_vocab[p])])        
    try:
       master_next=[]
       for z in y:
           for p in z:
               master_next.append([(new_vocab[p]),(new_vocab[p+1])])        
    except IndexError as error:
        pass
    master=master_next + master_prev
    dictionary=[]
    for ke in master:
        if ke not in dictionary:
            dictionary.append(ke)

    bigram=[]
    c=0
    for word in dictionary:
        for sec in master:
            if word==sec:
                c=c+1
        bigram.append([word,c])   
        c=0    
    bigram=sorted(bigram,key=lambda s:s[1],reverse=True) 
    bigram=[w for w in bigram if w[1]>1]
    
#tri-gram model    
    tri_prev=[]
    for z in y:
        for p in z:
            tri_prev.append([(new_vocab[p-2]),(new_vocab[p-1]),(new_vocab[p])])        
   
    try:
        tri_next=[]
        for z in y:
            for p in z:
                tri_next.append([(new_vocab[p]),(new_vocab[p+1]),(new_vocab[p+2])])       
    except IndexError as error:
        pass           

    tri=tri_next + tri_prev
    dictionary_=[]
    for lk in tri:
        if lk not in dictionary_:
            dictionary_.append(lk)

    trigram=[]
    ct=0
    for word in dictionary_:
        for sec in tri:
            if word==sec:
                ct=ct+1
        trigram.append([word,ct])   
        ct=0
                
    trigram=sorted(trigram,key=lambda s:s[1],reverse=True)    
    trigram=trigram[:10]
    
    return(trigram,bigram)        


#%%
#gensim sumarizer
def summary(stry):
    from gensim.summarization.summarizer import summarize
    summary=summarize(stry,word_count=100)
    return summary


#%%
# cosine similarity calculation
def cossim(doc1,doc2):
    from sklearn.metrics.pairwise import cosine_similarity as cs
    from sklearn.feature_extraction.text import CountVectorizer as cv

    x=[doc1,doc2]
    vectorizer=cv().fit_transform(x)
    vectors=vectorizer.toarray()    
    
    a=vectors[0].reshape(1,-1)
    b=vectors[1].reshape(1,-1)    
    
    similarity_score=cs(a,b)        
    
    return similarity_score

    

#%%
def eval_marks(inst_1,inst_2):

    summary_1=summary(inst_1)
    trigram_1,bigram_1=ngram(inst_1)
    
    summary_2=summary(inst_2)
    trigram_2,bigram_2=ngram(inst_2)
    
    
    aux_bigram_1=[]
    for i in bigram_1:
        aux_bigram_1.append(i[0])
    aux_bigram_2=[]
    for j in bigram_2:
        aux_bigram_2.append(j[0])
    bigram_match=np.in1d(aux_bigram_1,aux_bigram_2)
    num_match_bigram=(bigram_match==True).sum()

    clean_summary_1=preprocess(summary_1)
    clean_summary_2=preprocess(summary_2)
    
    score=cossim(clean_summary_1,clean_summary_2)
    score=score[0][0]
    
    #lst=[num_match_bigram,num_match_doc,num_match_summ,score]

    marks_sim=(score/0.61)*2.2 
    if marks_sim>3.5:
        marks_sim=3.5
    marks_bi=(num_match_bigram/16)*1.2
    if marks_bi>1.5:
        marks_bi=1.5
    marks=marks_sim+marks_bi
    marks=round(marks)
    print(marks)
    
    return marks    


# Task A
inst_5='Inheritance is a method of forming new classes using predefined classes. The new classes are called derived classes and they inherit the behaviours and attributes of the base classes. It was intended to allow existing code to be used again with minimal or no alteration. It also offers support for representation by categorization in computer languages; this is a powerful mechanism of information processing, vital to human learning by means of generalization and cognitive economy. Inheritance is occasionally referred to as generalization due to the fact that is-a relationships represent a hierarchy between classes of objects. Inheritance has the advantage of reducing the complexity of a program since modules with very similar interfaces can share lots of code. Due to this, inheritance has another view called polymorphism, where many sections of code are being controlled by some shared control code. Inheritance is normally achieved by overriding one or more methods exposed by ancestor, or by creating new methods on top of those exposed by an ancestor. Inheritance has a variety of uses. Each different use focuses on different properties, for example the external behaviour of objects, internal structure of an object, inheritance hierarchy structure, or software engineering properties of inheritance. Occasionally it is advantageous to differentiate between these uses, as it is not necessarily noticeable from context. '
inst_7='In object-oriented programming, inheritance is a way to form new classes (instances of which are called objects) using classes that have already been defined. The inheritance concept was invented in 1967 for Simula. The new classes, known as derived classes, take over (or inherit) attribute and behaviour of the pre-existing classes, which are referred to as base classes (or ancestor classes). It is intended to help reuse existing code with little or no modification. Inheritance provides the support for representation by categorization in computer languages. Categorization is a powerful mechanism number of information processing, crucial to human learning by means of generalization (what is known about specific entities is applied to a wider group given a belongs relation can be established) and cognitive economy (less information needs to be stored about each specific entity, only its particularities). Inheritance is also sometimes called generalization, because the is-a relationships represent a hierarchy between classes of objects. For instance, a "fruit" is a generalization of "apple", "orange", "mango" and many others. One can consider fruit to be an abstraction of apple, orange, etc. Conversely, since apples are fruit (i.e., an apple is-a fruit), apples may naturally inherit all the properties common to all fruit, such as being a fleshy container for the seed of a plant. An advantage of inheritance is that modules with sufficiently similar interfaces can share a lot of code, reducing the complexity of the program. Inheritance therefore has another view, a dual, called polymorphism, which describes many pieces of code being controlled by shared control code. Inheritance is typically accomplished either by overriding (replacing) one or more methods exposed by ancestor, or by adding new methods to those exposed by an ancestor.'
inst_10='The idea of inheritance in OOP refers to the formation of new classes with the already existing classes. The concept of inheritance was basically formulated for Simula in 1967. As a result, the newly created inherited or derived classes inherit the properties and behavior of the classes from which they are derived. These original classes are either called base classes or sometimes referred to as ancestor classes. The idea of inheritance is to reuse the existing code with little or no modification at all. The basic support provided by inheritance is that it represents by categorization in computer languages. The power mechanism number of information processing that is crucial to human learning by the means of generalization and cognitive economy is called categorization. Where generalization if the knowledge of specific entities and is applied to a wider group provided that belongs relation can be created. On the other hand cognitive economy is where less information needs to be stored about each specific entity except for some particularities. There are examples where we can have modules with similar interfaces. The advantage that inheritance provides is that it makes such modules share a lot of code which consequently reduces the complexity of the program.'
orig='In object-oriented programming, inheritance is a way to form new classes (instances of which are called objects) using classes that have already been defined. The inheritance concept was invented in 1967 for Simula. The new classes, known as derived classes, take over (or inherit) attributes and behavior of the pre-existing classes, which are referred to as base classes (or ancestor classes). It is intended to help reuse existing code with little or no modification. Inheritance provides the support for representation by categorization in computer languages. Categorization is a powerful mechanism number of information processing, crucial to human learning by means of generalization (what is known about specific entities is applied to a wider group given a belongs relation can be established) and cognitive economy (less information needs to be stored about each specific entity, only its particularities). Inheritance is also sometimes called generalization, because the is-a relationships represent a hierarchy between classes of objects. For instance, a "fruit" is a generalization of "apple", "orange", "mango" and many others. One can consider fruit to be an abstraction of apple, orange, etc. Conversely, since apples are fruit (i.e., an apple is-a fruit), apples may naturally inherit all the properties common to all fruit, such as being a fleshy container for the seed of a plant. An advantage of inheritance is that modules with sufficiently similar interfaces can share a lot of code, reducing the complexity of the program. Inheritance therefore has another view, a dual, called polymorphism, which describes many pieces of code being controlled by shared control code. Inheritance is typically accomplished either by overriding (replacing) one or more methods exposed by ancestor, or by adding new methods to those exposed by an ancestor. Complex inheritance, or inheritance used within a design that is not sufficiently mature, may lead to the Yo-yo problem.'














