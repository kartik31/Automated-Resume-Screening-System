
import numpy as np
import pandas as pd
import nltk,re
from nltk.corpus import stopwords


# In[2]:

import os,shutil
import sys
import logging
import six
import pdfminer.settings
pdfminer.settings.STRICT = False
import pdfminer.high_level
import pdfminer.layout
from pdfminer.image import ImageWriter


# In[3]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import glob
#import os
import warnings

import requests
from flask import (Flask, json, jsonify, redirect, render_template, request,
                   url_for)
from gensim.summarization import summarize
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from werkzeug import secure_filename

import pdf2txt as pdf

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

app = Flask(__name__)




app.config['UPLOAD_FOLDER'] = 'Original_Resumes/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf'])


class ResultElement:
    def __init__(self, rank, filename):
        self.rank = rank
        self.filename = filename
    # def printresult(self):
    #     print(str("Rank " + str(self.rank+1) + " :\t " + str(self.filename)))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def getfilepath(loc):
    temp = str(loc).split('\\')
    return temp[-1]

def deletefiles():
    print("s")


@app.route('/')
def index():
   return render_template('index.html')



@app.route('/uploadres', methods=['GET', 'POST'])
def uploadres():
    if request.method == 'POST':
        files = glob.glob('./Original_Resumes/*')
        for f in files:
            os.remove(f)

        uploaded_files = request.files.getlist("file[]")
      
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join('./Original_Resumes', filename))
        return render_template('index.html')
     
    return render_template('index.html')


@app.route('/uploaddes', methods=['GET', 'POST'])
def uploaddes():
    if request.method == 'POST':
        files = glob.glob('./Job_Description/*')
        for f in files:
            os.remove(f)

        file = request.files['file']
        file.save(os.path.join('./Job_Description', 'Job.txt'))
        return render_template('index.html')
     
    return render_template('index.html')
		




@app.route('/results')
def resume():
    Resume_Vector = []
    Ordered_list_Resume = []
    Ordered_list_Resume_Score = []
    LIST_OF_PDF_FILES = []
    for file in glob.glob("./Original_Resumes/*.pdf"):
        LIST_OF_PDF_FILES.append(file)


    files = glob.glob('./Parsed_Resumes/*')
    for f in files:
    	os.remove(f)


    print("Total Files to Parse\t" , len(LIST_OF_PDF_FILES))
    print("####### PARSING ########")
    for i in LIST_OF_PDF_FILES:
        pdf.extract_text([str(i)] , all_texts=None , output_type='text' , outfile='Parsed_Resumes\\'+getfilepath(i)+'.txt')

    print("Done Parsing.")


    Job_Desc = 0
    LIST_OF_TXT_FILES = []
    for file in glob.glob("./Job_Description/*.txt"):
        LIST_OF_TXT_FILES.append(file)

    for i in LIST_OF_TXT_FILES:
        f = open(i , 'r')
        text = f.read()
        
        tttt = str(text)
        print(tttt)
        tttt = summarize(tttt, word_count=100)
        print(tttt) ## Summarized Text
        text = [tttt]
        f.close()
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit(text)
        vector = vectorizer.transform(text)
        # print(vector.shape)
        # print(vector.toarray())
        Job_Desc = vector.toarray()
        print("\n\n")

    
    LIST_OF_TXT_FILES = []
    for file in glob.glob("./Parsed_Resumes/*.txt"):
        LIST_OF_TXT_FILES.append(file)

    for i in LIST_OF_TXT_FILES:
        Ordered_list_Resume.append(i)
        f = open(i , 'r')
        text = f.read()
        
        tttt = str(text)
        tttt = summarize(tttt, word_count=100)
        # print(tttt) ## Summarized Text
        text = [tttt]
        f.close()

        # vectorizer = TfidfVectorizer(stop_words='english')
        # vectorizer.fit(text)
        vector = vectorizer.transform(text)
        # print(vector.shape)
        # print(vector.toarray())
        aaa = vector.toarray()
        Resume_Vector.append(vector.toarray())
        # print("\n\n")




    # print("##############################################")
    # print(Resume_Vector)
    # print("##############################################")


    for i in Resume_Vector:
        # print("This is a single resume" , i)

        samples = i
        
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(samples) 
        NearestNeighbors(algorithm='auto', leaf_size=30)
        # print(Resume_Vector[0])
        # print(Job_Desc)
        # print(neigh.kneighbors(Job_Desc)[0]) 
        Ordered_list_Resume_Score.extend(neigh.kneighbors(Job_Desc)[0][0].tolist())
        
        # return '''
        # <!doctype html>
        # <title>Upload new File</title>
        # <h1>Upload new File</h1>
        # <h3>{Job_Desc}</h3>
        # '''
    #Z = [x for _,x in sorted(zip(Ordered_list_Resume_Score,Ordered_list_Resume))]
    #print(Ordered_list_Resume)
    #print(Ordered_list_Resume_Score)
    flask_return = []
    
    stoplist = stopwords.words('english')
    stoplist.append('\n')
    skill=open('Job_Description/Job.txt','r')
    dir='textresume'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    files_no_ext = [".".join(f.split(".")[:-1]) for f in os.listdir('Original_Resumes')]
    print(files_no_ext)
    print("knn")
    print(Ordered_list_Resume_Score)
    for f in files_no_ext:
        a=open('textresume/'+f+'.txt','a')
        a.close()
    resume_pdf=os.listdir('Original_Resumes')
    resume_txt=os.listdir('textresume')
    def extract_text(files=[], outfile=[],
            _py2_no_more_posargs=None,  # Bloody Python2 needs a shim
            no_laparams=False, all_texts=None, detect_vertical=None, # LAParams
            word_margin=None, char_margin=None, line_margin=None, boxes_flow=None, # LAParams
            output_type='text', codec='utf-8', strip_control=False,
            maxpages=0, page_numbers=None, password="", scale=1.0, rotation=0,
            layoutmode='normal', output_dir=None, debug=False,
            disable_caching=False, **other):
        if _py2_no_more_posargs is not None:
            raise ValueError("Too many positional arguments passed.")
        """if not files:
            raise ValueError("Must provide files to work upon!")"""

        # If any LAParams group arguments were passed, create an LAParams object and
        # populate with given args. Otherwise, set it to None.
        if not no_laparams:
            laparams = pdfminer.layout.LAParams()
            for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
                paramv = locals().get(param, None)
                if paramv is not None:
                    setattr(laparams, param, paramv)
        else:
            laparams = None

        imagewriter = None
        if output_dir:
            imagewriter = ImageWriter(output_dir)

        """if output_type == "text" and outfile != "-":
            for override, alttype in (  (".htm", "html"),
                                    (".html", "html"),
                                    (".xml", "xml"),
                                    (".tag", "tag"),
                                 (".txt","text")):
            if outfile.endswith(override):
                output_type = alttype"""

        if outfile == []:
            outfp = sys.stdout
            if outfp.encoding is not None:
                codec = 'utf-8'
        else:
            i=0
            for outfi in outfile:
                fname=files[i]
                i+=1
                outfp = open('textresume/'+outfi, "w")
            
                with open('Original_Resumes/'+fname, "rb") as fp:
                    pdfminer.high_level.extract_text_to_fp(fp, **locals())
        return

    output=extract_text(resume_pdf,resume_txt)
    for f in resume_txt:
        file=open('textresume/'+f,'r+')
        data=file.read()
        data=re.sub(r'[^\x00-\x7F]+',' ', data)
        data=data.replace('\n',' ')
        file.seek(0)
        file.write(data)
    skill.seek(0)
    tfv=TfidfVectorizer(token_pattern = r"(?u)\b\w+\b",stop_words=stoplist)
    tfv.fit(skill) 
    skill.seek(0)
    y=tfv.transform(skill)
    skill.seek(0)
    df2=pd.DataFrame(columns=tfv.get_feature_names())
    s2=pd.DataFrame(y.toarray(), columns=tfv.get_feature_names())
    for f in os.listdir('textresume'):
        file = open('textresume/'+f,'r')    
        file.seek(0)
        y=tfv.transform(file)
        x=y.toarray().sum(axis=0)
        df2.loc[len(df2)]=x
    #print df2
    li=[]
    for i in range(0,len(df2)):
        li.append((s2.loc[0]*df2.loc[i]).sum())
    w=[]
    for q in range(0,len(li)):
        ff=Ordered_list_Resume_Score[q]-li[q]
        w.append(ff)
    Z = [x for _,x in sorted(zip(w,Ordered_list_Resume))]
    #print(w)
    rating=dict(zip(os.listdir('Original_Resumes'),li))

    rating=sorted(rating.items(), key=lambda x:x[1])
    rating=rating[::-1]
    #print rating
    print("content")
    print(li)
    print("ensemble")
    print(w)
    for n,i in enumerate(Z):
        # print("Rank\t" , n+1, ":\t" , i)
        # flask_return.append(str("Rank\t" , n+1, ":\t" , i))
        name = getfilepath(i)
        name = name.split('.')[0]
        rank = n+1
        res = ResultElement(rank, name)
        flask_return.append(res)
        # res.printresult()
        #print(f"Rank{res.rank+1} :\t {res.filename}")
    return render_template('result.html', results = flask_return)

if __name__ == '__main__':
   # app.run(debug = True) 
    app.run('0.0.0.0' , 5000 , debug=True, threaded=True)
