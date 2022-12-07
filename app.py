       # -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:27:07 2021

@author: PC
"""

from __future__ import division, print_function
from flask import Flask,render_template,url_for,request, send_file
from flask_bootstrap import Bootstrap
import os

from flask_mysqldb import MySQL
import yaml

from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import csv
import subprocess
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


app = Flask(__name__) 
Bootstrap(app)

db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
mysql = MySQL(app)



@app.route('/')
def index():
	
    
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    df_datas = pd.read_csv('senate.csv')
    senate_data = df_datas[['MATTER','SENATE DECISION','Category']]
    df_datas['Date'] = pd.to_datetime(df_datas['Date'].apply(lambda x: x.split()[0]))
    import matplotlib.pyplot as plt
    
    new_data = df_datas[['MATTER','SENATE DECISION']]

    import sys
    import string
    from nltk.corpus import stopwords
    import matplotlib.pyplot as plt
    from nltk.stem import WordNetLemmatizer
    import warnings
    warnings.filterwarnings('ignore')
    import io
    from io import StringIO


    senate_data['MATTER']= senate_data['MATTER'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    senate_data['MATTER']= senate_data['MATTER'].str.replace('[^\w\s]','')
    senate_data['MATTER']= senate_data['MATTER'].str.replace('\d+', '') # remove numeric values from between the words
    senate_data['MATTER']= senate_data['MATTER'].apply(lambda x: x.translate(string.digits))
    stop = stopwords.words('english')
    stemmer = WordNetLemmatizer()
    senate_data['MATTER']= [stemmer.lemmatize(word) for word in senate_data['MATTER']]
    senate_data['MATTER']= senate_data['MATTER'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


    senate_data['SENATE DECISION']= senate_data['SENATE DECISION'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    senate_data['SENATE DECISION']= senate_data['SENATE DECISION'].str.replace('[^\w\s]','')
    senate_data['SENATE DECISION']= senate_data['SENATE DECISION'].str.replace('\d+', '') # remove numeric values from between the words
    senate_data['SENATE DECISION']= senate_data['SENATE DECISION'].apply(lambda x: x.translate(string.digits))
    stop = stopwords.words('english')
    stemmer = WordNetLemmatizer()
    senate_data['SENATE DECISION']= [stemmer.lemmatize(word) for word in senate_data['SENATE DECISION']]
    senate_data['SENATE DECISION']= senate_data['SENATE DECISION'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    df = senate_data[['MATTER','Category']]
    df = df[pd.notnull(df['MATTER'])]
    col = ['Category', 'MATTER']
    df = df[col]
    df.columns
    df.columns = ['Category', 'Text_Data']
    
    df['category_id'] = df['Category'].factorize()[0]
    from io import StringIO
    category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Category']].values)
    print(df.head())
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    
    features = tfidf.fit_transform(df.Text_Data).toarray()
    labels = df.category_id
    print(features.shape)
    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer


    X_train, X_test, y_train, y_test = train_test_split(df['Text_Data'], df['Category'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
        
    if request.method == 'POST':
        
        file = request.files['image']
        select = request.form['select']
        
        
        basepath = os.path.dirname(__file__)
        
        file_path = os.path.join(
            
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        
        from docx.api import Document
        
        document = Document(file)
        table = document.tables[1]
        
        data = []
        
        keys = None
        for i, row in enumerate(table.rows):
            text = (cell.text for cell in row.cells)
        
            if i == 0:
                keys = tuple(text)
                continue
            row_data = dict(zip(keys, text))
            data.append(row_data)
            
        
        
        df_test = pd.DataFrame(data)
        df_test = df_test[['MATTER','SENATE DECISION']]
        df_test = df_test[df_test.MATTER != 'MATTER']
        
        df_test['MATTER']= df_test['MATTER'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        df_test['MATTER']= df_test['MATTER'].str.replace('[^\w\s]','')
        df_test['MATTER']= df_test['MATTER'].str.replace('\d+', '') # remove numeric values from between the words
        df_test['MATTER']= df_test['MATTER'].apply(lambda x: x.translate(string.digits))
        stop = stopwords.words('english')
        stemmer = WordNetLemmatizer()
        df_test['MATTER']= [stemmer.lemmatize(word) for word in df_test['MATTER']]
        df_test['MATTER']= df_test['MATTER'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


        df_test['SENATE DECISION']= df_test['SENATE DECISION'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        df_test['SENATE DECISION']= df_test['SENATE DECISION'].str.replace('[^\w\s]','')
        df_test['SENATE DECISION']= df_test['SENATE DECISION'].str.replace('\d+', '') # remove numeric values from between the words
        df_test['SENATE DECISION']= df_test['SENATE DECISION'].apply(lambda x: x.translate(string.digits))
        stop = stopwords.words('english')
        stemmer = WordNetLemmatizer()
        df_test['SENATE DECISION']= [stemmer.lemmatize(word) for word in df_test['SENATE DECISION']]
        df_test['SENATE DECISION']= df_test['SENATE DECISION'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        
        
        x_test = df_test['MATTER']
        inputs = count_vect.transform(x_test)
        
        savedmodel = pickle.load(open('senate_data.h5','rb'))
        
        data = savedmodel.predict(inputs)
        
        pred = pd.DataFrame(data, columns=['Categorized_Result'])
        final_pred = pd.concat([new_data, pred], axis=1, join='inner')
        final_pred1 = final_pred.loc[final_pred['Categorized_Result']== select]

        
        
        
        import matplotlib.pyplot as plt
        fg, ax =plt.subplots(figsize=(12,7))
        ax.plot(df_datas['Date'], df_datas['Category']==select, label='Category',color='green')
        ax.set_xlabel('Date',size=15)
        ax.set_ylabel('Number of Occurence',size=15)
        ax.legend()
        plt.savefig('static/img/new.png')
        
         
        
       

    return render_template('predict.html', final_pred1=final_pred1.to_html(), select=select)

@app.route('/add_document')
def add_document():
    
    
    cur1 = mysql.connection.cursor()
        
    result1 = cur1.execute("SELECT * from senate_data")
    if(result1>0):
        
    
        view = cur1.fetchall()
    
    if request.method == 'POST':
        
        file = request.files['file']
            
        basepath = os.path.dirname(__file__)
        filename = secure_filename(file.filename)
        
        file_path = os.path.join(
            
            basepath, '', filename)

        file.save(file_path)
        

        
        cur = mysql.connection.cursor()
        query = "INSERT INTO senate_data (meeting) VALUES (%s)"
        cur.execute(query, (filename, ))
        mysql.connection.commit()
        cur.close()
        marked = 'sucessful'
        
        
       
            
            
            

        return render_template('add.html', marked=marked, view=view)
    
    return render_template('add.html')


@app.route('/document', methods=["POST"])

def document():
    
    
    if request.method == 'POST':
        
        file = request.files['file']
            
        basepath = os.path.dirname(__file__)
        filename = secure_filename(file.filename)
        
        file_path = os.path.join(
            
            basepath, '', filename)

        file.save(file_path)
        

        
        cur = mysql.connection.cursor()
        query = "INSERT INTO senate_data (meeting) VALUES (%s)"
        cur.execute(query, (filename, ))
        mysql.connection.commit()
        cur.close()
        marked = 'sucessful'
        
        
        cur1 = mysql.connection.cursor()
        
        result1 = cur1.execute("SELECT * from senate_data")
        if(result1>0):
        
            view = cur1.fetchall()
            
            
            

        return render_template('add.html', marked=marked, view=view)
        
    return render_template('add.html')

@app.route('/view', methods=['POST'])



def view():

    if request.method == 'POST':
        cur1 = mysql.connection.cursor()
        
        result1 = cur1.execute("SELECT * from senate_data")
        if(result1>0):
        
            view = cur1.fetchall()
            

        doc = request.form['tochi']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
		'uploads', doc)
        
        
        p = file_path

        
       
        subprocess.Popen([file_path], shell=True)
        
        
       
    return send_file(p, as_attachment=True)
    return render_template('add.html',view=view)



@app.route('/view_document')

def view_document():

    cur1 = mysql.connection.cursor()
        
    result1 = cur1.execute("SELECT * from senate_data")
    if(result1>0):
        view = cur1.fetchall()
            
        return render_template('add.html', view=view)

if __name__=='__main__':
	
	app.run(debug=True)
