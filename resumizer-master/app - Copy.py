"""
Created on March 2018
@author: 
# ▒█▀▀█ █▀▀█ █▀▀▄ █▀▀ ▒█▀▀█ █░░█ ▀▀█▀▀ █▀▀ 
# ▒█░░░ █░░█ █░░█ █▀▀ ▒█▀▀▄ █▄▄█ ░░█░░ █▀▀ 
# ▒█▄▄█ ▀▀▀▀ ▀▀▀░ ▀▀▀ ▒█▄▄█ ▄▄▄█ ░░▀░░ ▀▀▀ 
"""

#################################################################################################################################################




# ▒█▀▀█ ▒█▀▀▀ ▒█▀▀▀█ ▒█░▒█ ▒█▀▄▀█ ▒█▀▀▀ 　 ▒█▀▀▀█ ▒█▀▀█ ▒█▀▀█ ▒█▀▀▀ ▒█▀▀▀ ▒█▄░▒█ ▀█▀ ▒█▄░▒█ ▒█▀▀█ 　 ▒█▀▀▀█ ▒█░░▒█ ▒█▀▀▀█ ▀▀█▀▀ ▒█▀▀▀ ▒█▀▄▀█ 
# ▒█▄▄▀ ▒█▀▀▀ ░▀▀▀▄▄ ▒█░▒█ ▒█▒█▒█ ▒█▀▀▀ 　 ░▀▀▀▄▄ ▒█░░░ ▒█▄▄▀ ▒█▀▀▀ ▒█▀▀▀ ▒█▒█▒█ ▒█░ ▒█▒█▒█ ▒█░▄▄ 　 ░▀▀▀▄▄ ▒█▄▄▄█ ░▀▀▀▄▄ ░▒█░░ ▒█▀▀▀ ▒█▒█▒█ 
# ▒█░▒█ ▒█▄▄▄ ▒█▄▄▄█ ░▀▄▄▀ ▒█░░▒█ ▒█▄▄▄ 　 ▒█▄▄▄█ ▒█▄▄█ ▒█░▒█ ▒█▄▄▄ ▒█▄▄▄ ▒█░░▀█ ▄█▄ ▒█░░▀█ ▒█▄▄█ 　 ▒█▄▄▄█ ░░▒█░░ ▒█▄▄▄█ ░▒█░░ ▒█▄▄▄ ▒█░░▒█ 


"""
●   A web app to help employers by analysing resumes and CVs, surfacing candidates that best match the position and filtering out those who don't.
●   Used recommendation engine techniques such as KNN, content based filtering for fuzzy matching job description with multiple resumes.
Prerequisites
    Gensim
    Numpy==1.11.3
    Pandas
    Sklearn
    Dash
Tested on Ubuntu 16.04 LTS amd64 xenial image built on 2017-09-19
        To Run this code: 
            # python app.py
        Runs on localhost:5000
"""

import glob
import os
import warnings

import requests
from flask import (Flask, json, jsonify, redirect, render_template, request,
                   url_for)
from gensim.summarization import summarize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
    temp = str(loc).split('/')
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
        pdf.extract_text([str(i)] , all_texts=None , output_type='text' , outfile='./Parsed_Resumes/'+getfilepath(i)+'.txt')

    print("Done Parsing.")


    Job_Desc = 0
    LIST_OF_TXT_FILES = []
    for file in glob.glob("./Job_Description/*.txt"):
        LIST_OF_TXT_FILES.append(file)

    for i in LIST_OF_TXT_FILES:
        f = open(i , 'r')
        text = f.read()
        
        tttt = str(text)
        # print(tttt)
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
    Z = [x for _,x in sorted(zip(Ordered_list_Resume_Score,Ordered_list_Resume))]
    print(Ordered_list_Resume)
    print(Ordered_list_Resume_Score)
    flask_return = []
    for n,i in enumerate(Z):
        # print("Rank\t" , n+1, ":\t" , i)
        # flask_return.append(str("Rank\t" , n+1, ":\t" , i))
        name = getfilepath(i)
        name = name.split('.')[0]
        rank = n+1
        res = ResultElement(rank, name)
        flask_return.append(res)
        # res.printresult()
        print(f"Rank{res.rank+1} :\t {res.filename}")
    return render_template('result.html', results = flask_return)

if __name__ == '__main__':
   # app.run(debug = True) 
    app.run('0.0.0.0' , 5000 , debug=True, threaded=True)
