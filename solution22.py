"""
 this program finds the distance between user question and the available questions and answers , in order to find the most matching question

Semantic similarity
 1. Sentence Segmentation: we split the input string/question string and answers strings into sentences. and further sentences into words.
 2. Abbreviations of acronyms: we expand the captilized possible acronyms with abbreviations given by  a file(online).
 3. Stop words elimination: we eliminate the stop from the word vectors.
 4. Remove Punctuation: we remove punctuations.
 5. Parts of speech Tagging: the words are tagged to their corresponding parts of english sentence grammar.
 6. Word sense Identification: we find the word sense of the words using wordnet package. it gives the synonyms, and definitions of the word senses 
 7. Word Sense Disambiguation using Lesk algorithm: we capture the most appropriate word sense applicable from the synonyms provided by wordnet.
 8. Path similarity between word senses: word senses are stored in a paths representative of their etymology including hypernyms and hyponyms of the word.Path similarity gives a notion of their closeness.
 9. lch_similairty between the word senses: Leacock-Chodorow Similarity:  a score denoting how similar two word senses are, based on the shortest path that connects the senses 
10. wup_similarity between the word senses: Wu-Palmer Similarity: how similar two word senses are, based on the depth of the two senses in the taxonomy
11. we find the max similarity between any words of two strings and add max similarity for all words in from the strings

Vector similarity:
 1. we obtain vectors out of the strings using their term-document-frequency and inverse document-frequency score (tdf-idf)
 2. we find mathematical similarity between those two vectors using the following scores
     a. Cosine similarity: the cosine distance between vector A and B based on the angle between them.
     b. Correlation Coefficient: indicates how positiviley or negatively correlated are the vectors.
     c. Bray-Curtis Score: Quantify the compositional dissimilarity between two different sites, based on counts at each site.
     d. city-Block: Manhattan score: The distance between two points measured along axes at right angles
     e. Sorenson-dice coefficient based on bigrams
     f. yule distance between vectors: yule dissimilarity score between two boolean arrays
     g. hamming distance between vectors: number used to denote the difference between two binary strings

we combine scores using arbitrary weights to produce a total similarity score.
based on which the appropriate question and the appropriate answer are obtained.
"""
import json
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import nltk as nltk
import re
import numpy as np
import numpy.linalg as LA
import scipy.spatial.distance as dist

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score  
#def 
def abbrevate(given):
    sentence=given.split()
    abbr_dict={}
    foldr="C:/Users/Yogesh/Desktop/New"
    fil='abbr.txt'
    with open(flodr+'/'+fil) as abbr_file:
        for line in abbr_file:
                line_words=line.split('\t');
                #print (line_words)
                abbr_dict[line_words[0]]=line_words[1]

        for w in range(len(sentence)):
                caps=1
                for ch in sentence[w]:
                        if(ch.islower()):
                                caps=0
                if(caps==1 and sentence[w] in abbr_dict):
                       # print(sentence1[w]+"has caps and "+abbr_dict[sentence1[w]]);
                       # print(string1.count(sentence1[w]));
                        given=given.replace(sentence[w],abbr_dict[sentence[w]],1)
    return given

def similarity(string1,string2):

        #split the string into sentences and sentences into words
        sentences1=[d for d in re.split('\.\W',string1)]
        sentences2=[d for d in re.split('\.\W',string2)]

        #sentences1=[d for d in string1.split('|')]
        #sentences2=[d for d in string2.split('|')]

        sentences1=[d.split('|') for d in sentences1]
        sentences1=[d for e in sentences1 for d in e]

        sentences2=[d.split('|') for d in sentences2]
        sentences2=[d for e in sentences2 for d in e]
        
        #remove stop words
        stop=stopwords.words('english')

        #clean_sentence=[ j for j in sentence if j not in stop]
        #sentence1=[j for j in sentence1 if j not in stop]
        #sentence2=[j for j in sentence2 if j not in stop]

        #remove Punctuation
        regex=re.compile('[%s]' % re.escape(string.punctuation))
        #remove_punct_map=dict((ord(char),None) for char in string.punctuation)
        #regex.sub('', s)
        #clean_sentence_npunct=[c.translate(remove_punct_map) for c in clean_sentence]

        sentences1=[regex.sub('', c) for c in sentences1]
        sentences2=[regex.sub('', c) for c in sentences2]

        sentences1=[c.replace('\n','') for c in sentences1]
        sentences2=[c.replace('\n','') for c in sentences2]

        sentences1=[c.replace('\t','') for c in sentences1]
        sentences2=[c.replace('\t','') for c in sentences2]

        
        #print(sentences1)
        #print(sentences2)

        totalsimilarity=0;
        for sent1 in sentences1:

             sentence1=sent1.split()

             for k,ktag in nltk.pos_tag(sentence1):

                maxsimilarity=0

                syn1=lesk(sentence1,k)

                #print(k+" and"+ktag)
                if syn1 is None:
                        #print("syn1 is none");
                        for i,sym1  in enumerate(wn.synsets(k)):
                                syn1=sym1
                if syn1 is not None:
                    
                     for sent2 in sentences2:
                         
                         sentence2=sent2.split()
                         
                         for j,jtag in nltk.pos_tag(sentence2):
                             
                             #  print(syn1.name()+" and "+syn2.name())
                               sim=0
                               a=0
                                #print(j+ " and " +jtag)
                             #   if(ktag is jtag):
                               if(a==0):
                                        syn2=lesk(sentence2,j)
                                        #print(syn1.name()+" and "+syn2.name())
                                        if syn2 is None:
                                                #print("syn2 is none")
                                                for i,sym2 in enumerate(wn.synsets(j)):
                                                        syn2=sym2
                                        if syn2 is not None:
                                                #print(j+"is none")
                                        #else:
                        #                        print(syn1.name()+" and "+syn2.name())
                                                ps=syn1.path_similarity(syn2)
                                                ws=syn1.wup_similarity(syn2)
                                                ls=0
                                                if(syn1.name().split('.')[1] == syn2.name().split('.')[1]):
                         #                               print(syn1.name()+" and "+syn2.name()) 
                                                        ls=syn1.lch_similarity(syn2)
                                                if ps is not None:
                                                        sim=sim+ps
                                                if ws is not None:
                                                        sim=sim+ws
                                                if ls is not None:
                                                        sim=sim+ls
                                                if maxsimilarity < sim :  
                                                        maxsimilarity=sim
        totalsimilarity=totalsimilarity+maxsimilarity
        #print ("\nscore"+str(totalsimilarity));
        return totalsimilarity


data=[]
abbr_dict={}
with open(foldr+"/data.json") as data_file:
	for line in data_file:
		data.append(json.loads(line))
#"""
#abbreviations data
with open(foldr+"/abbr.txt") as abbr_file:
        for line in abbr_file:
                line_words=line.split('\t');
                #print (line_words)
                abbr_dict[line_words[0]]=line_words[1]
        #print (line_words)
#"""        
#remove stop words
		
stop=stopwords.words('english')
words=[]
counts=[]
worddict={}

userinput="howb to book a hotel using gocash. is it free?" 
#Strinput=Strinput.split(

maxscore=0;
maxquestion=0;
Qvectorizer=CountVectorizer(stop_words=stop)
Avectorizer=CountVectorizer(stop_words=stop)
QTvectorizer=TfidfVectorizer()
ATvectorizer=TfidfVectorizer()

all_questions=[d['question'] for d in data]
all_questions=[userinput]+all_questions
all_answers=[d['answer'] for d in data]
all_answers=[userinput]+all_answers

QuestionTVectorArray=QTvectorizer.fit_transform(all_questions)
AnswerTVectorArray=ATvectorizer.fit_transform(all_answers)

#print "question cosine similairity-->",cosine_similarity(QuestionTVectorArray[0:1],QuestionTVectorArray)
#print "answer cosine similarity-->",cosine_similarity(AnswerTVectorArray[0:1],AnswerTVectorArray)
Qcosines=cosine_similarity(QuestionTVectorArray[0:1],QuestionTVectorArray)
Acosines=cosine_similarity(AnswerTVectorArray[0:1],AnswerTVectorArray)

Qbray=[dist.braycurtis(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Abray=[dist.braycurtis(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

Qcanberra=[dist.canberra(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Acanberra=[dist.canberra(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

Qhamming=[dist.hamming(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Ahamming=[dist.hamming(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

Qcorrelation=[dist.correlation(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Acorrelation=[dist.correlation(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

Qcityblock=[dist.cityblock(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Acityblock=[dist.cityblock(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

Qdice=[dist.dice(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Adice=[dist.dice(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

Qyule=[dist.yule(QuestionTVectorArray[0].toarray(),u.toarray()) for u in QuestionTVectorArray]
Ayule=[dist.yule(AnswerTVectorArray[0].toarray(),u.toarray()) for u in AnswerTVectorArray]

#C_Q=np.histogram2d(QuestionTVectorArray[1],QuestionTVectorArray[1])[0]

#print "question mutual info-->",mutual_info_score(None,None,contigency=C_Q)#QuestionTVectorArray[0:1],QuestionTVectorArray)
#QuestionVectorArray=Qvectorizer.fit_transform(all_questions).toarray()
#AnswerVectorArray=Avectorizer.fit_transform(all_answers).toarray()

#QUserinputVectorArray=Qvectorizer.transform(userinput).toarray()
#AUserinputVectorArray=Avectorizer.transform(userinput).toarray()
#cx=lambda a,b:round(np.inner(a,b)/(LA.norm(a)*LA.norm(b)),3)
"""
mincosine=1
minques=0
for Qv in range(len(QuestionVectorArray)):
   # print "Question vector:",QuestionVectorArray[Qv]
   # print "Answer vector:",AnswerVectorArray[Qv]
   # print "input vector:",AUserinputVectorArray[0]
    for Quservector in QUserinputVectorArray:
        Qcosine=cx(QuestionVectorArray[Qv],Quservector)
        Acosine=cx(AnswerVectorArray[Qv],AUserinputVectorArray[0])
        print Acosine;
        print Qcosine;
        sumcosine=Qcosine+Acosine
        if(mincosine>sumcosine):
            mincosine=sumcosine
            minques=Qv
print "best cosine",mincosine,"best question",data[minques]['question'],"best answer",data[minques]['answer']
#"""
for i1 in range(len(data)):
    sentence=data[i1]["question"].split()
    string1=userinput;
    string2=data[i1]["question"]
    string3=data[i1]["answer"]
 
    #string3="which is the AAA largest democracy in the world"

    string1=abbrevate(string1)
    string2=abbrevate(string2) 
    string3=abbrevate(string3)
    
    #score13=similarity(string1,string3)
  #  score12=similarity(string1,string2)
    questionscore=similarity(string1,string2)
    answerscore=similarity(string1,string3)
    
    #print(str(score13)+"is the score")
    #print("question score:"+str(questionscore)+"  answer score: "+str(answerscore))
    totalsimilarity=0
     #add similarity scores
    totalsimilarity=75*(questionscore)+150*(answerscore)
     #subtract bray score,canberra score
    totalsimilarity=totalsimilarity+15*(Qbray[i1+1]+Abray[i1+1])+15*(Qcanberra[i1+1]+Acanberra[i1+1])+15*(Qhamming[i1+1]+Ahamming[i1+1])+10*(Qyule[i1+1]+Ayule[i1+1])
     # Add Dice score,correlation score,Yule score
    totalsimilarity=totalsimilarity+75*(Qcorrelation[i1+1]+Acorrelation[i1+1])+55*(Qcosines[0][i1]+Acosines[0][i1])+50*(Qdice[i1+1]+Adice[i1+1])
    print("Total: ",totalsimilarity," cosines:",Qcosines[0][i1]," correlation:",Qcorrelation[i1+1]," bray:",Qbray[i1+1]," canberra:",Qcanberra[i1+1]," hamming:",Qhamming[i1+1]," yule:",Qyule[i1+1]," dice:",Qdice[i1+1])
    #sentence1=string1.split()
    #sentence2=string2.split()
    #sentence3=string3.split()
    #print(string1+" and "+string2+" and "+string3
    
    if(maxscore<totalsimilarity):#-15*(Qbray[i1+1]-Abray[i1+1])-15*(Qcanberra[i1+1]+Acanberra[i1+1])-15*(Qhamming[i1+1]+Ahamming[i1+1])-10*(Qyule[i1+1]+Ayule[i1+1])):
                maxscore=totalsimilarity#-15*(Qbray[i1+1]-Abray[i1+1])-15*(Qcanberra[i1+1]+Acanberra[i1+1])-15*(Qhamming[i1+1]+Ahamming[i1+1])-10*(Qyule[i1+1]+Ayule[i1+1])
                maxquestion=i1
print("\n\n\ngiven question:"+userinput+"\nthe best question\n question:"+data[maxquestion]["question"]+" \nBest answer \nanswer:"+data[maxquestion]["answer"]+"\n score:"+str(maxscore));
"""
#Lemmatize words
                       
Lemmatizer=WordNetLemmatizer()
synonyms=[]

clean_sentence_np_lemmatized=[Lemmatizer.lemmatize(c) for c in clean_sentence_npunct]

clean_sentence_np_lm_related=[]
synonyms=[]

for c in clean_sentence_np_lemmatized:
    
    for i,syn in enumerate(wn.synsets(c)):

        synonyms=syn.lemma_names()
        synonyms=[n.replace('_','') for n in synonyms]
        synonyms=["R-"+n for n in synonyms]
        synonyms.append(c)
        
    clean_sentence_np_lm_related=clean_sentence_np_lm_related+synonyms
    
for k in clean_sentence_np_lm_related:
    worddict[k]=clean_sentence_np_lm_related.count(k)

print(clean_sentence)
print(worddict)
print(totalsimilarity)

#"""

