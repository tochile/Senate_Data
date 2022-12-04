       # -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:27:07 2021

@author: PC
"""

from __future__ import division, print_function
from flask import Flask,render_template,url_for,request
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
nltk.download('stopwords')

app = Flask(__name__) 
Bootstrap(app)


	

@app.route('/')
def index():
    
    return render_template('predict.html')



if __name__=='__main__':
	
	app.run(debug=True)
