# A very simple Flask Hello World app for you to get started with...
from flask import Flask,render_template,request
import pandas as pd
from googletrans import Translator
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import numpy as np
from translate import Translator
import nltk
#model=pd.read_pickle('/home/hemsingh121/mysite/Salary_mdl.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('LnB1.html')

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=' '.join(final_summary)
    return summary

@app.route('/prediction',methods=['GET','POST'])
def predict():
    /*
    f=[x for x in request.form.values()]
    list1 = []
    list1 = f.split('।')
    translation=[]
    translator= Translator(from_lang="hi",to_lang="en")
    for i in range(len(list1)):
       translation.append(translator.translate(list1[i]))
    #print(translation)
    translation=str(' '.join(translation))
    */
    if request.method=='POST':
        translation=(request.form["nm"])
    
    summ=(summarize(translation, 0.3))
    /*
    from translate import Translator
    list1 = []
    list1 = summ.split(',')
    translation1=[]
    translator= Translator(from_lang="en",to_lang="hi")
    for i in range(len(list1)):
       translation1.append(translator.translate(list1[i]))
    #print(list[i])
    #print("\n")

    translation1=str(' '.join(translation1))
    return render_template('LnB.html',prediction_text=str(translation1))
*/
    return render_template('LnB.html',prediction_text=str(summ))


