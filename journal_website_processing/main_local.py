# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
get features for each journal
"""


import cv2
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import font_extractor_local
import cssutils
import requests
import logging
cssutils.log.setLevel(logging.CRITICAL)
from urllib.parse import urlparse


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import website_content_extractor_local
from joblib import Parallel, delayed
import pickle,html_tag_extractor_local
import shutil
import random



df = pd.read_csv('doaj_journal_list_urls_from_doaj_bing_v1.csv')

unwhite_path = './screenshots/out_DOAJ/'
white_path = './screenshots/in_DOAJ/'
from pathlib import Path
step = 100

#all_urls = list(df['URL'])

tmp = df[df['Reason'] != 'Still in DOAJ index']
all_urls = list(tmp['ISSN'].apply(str.strip))

tmp2 = df[df['Reason'] == 'Still in DOAJ index']
all_urls2 = list(tmp2['ISSN'].apply(str.strip))


request = ['about','aims','editor','ethics policy','open access policy','copyright policy']
info_key_terms, entity_recognizer, nlp, univ_rank = website_content_extractor_local.initilize_web_content_extractor_local()

span = int((len(all_urls)+step-(len(all_urls)%step))/step)


span2 = int((len(all_urls2)+step-(len(all_urls2)%step))/step)


#chrome_path = './tools/chromedriver'


def get_tag_feature_local(urls,file_name,feature_dir,html_dir,jobs):
    web_html_tag_feature = []
    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]
        #web_html_tag_feature += Parallel(n_jobs=jobs, verbose=10,timeout=300)(delayed(html_tag_extractor_local.get_html_tag_local)(html_dir,url) for url in urls_compute)

        try:
            res = Parallel(n_jobs=jobs, verbose=10,timeout=300)(delayed(html_tag_extractor_local.get_html_tag_local)(html_dir,url) for url in urls_compute)
            web_html_tag_feature += res
            print(res)
        except:
            print('go divide')
            for ur in urls_compute:
                try:
                    res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(html_tag_extractor_local.get_html_tag_local)(html_dir,url) for url in [ur])
                    web_html_tag_feature += res
                except:
                    print('cannot find tag')
                    web_html_tag_feature.append([])

    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_html_tag_feature, f)



def get_font_feature_local(urls,file_name,feature_dir,html_dir,jobs):
    web_font_feature = []
    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]

        try:
            res = Parallel(n_jobs=jobs, verbose=10,timeout=300)(delayed(font_extractor_local.get_font_local)(html_dir,url) for url in urls_compute)
            web_font_feature += res
            print(res)
        except:
            print('go divide')
            for ur in urls_compute:
                try:
                    res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(font_extractor_local.get_font_local)(html_dir,url) for url in [ur])
                    web_font_feature += res
                except:
                    print('cannot find font')
                    web_font_feature.append([])

    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_font_feature, f)





def get_content_feature_local(urls,file_name,jobs,feature_dir,html_dir):

    web_content = []
    abstract_data = []
    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]

        try:
            res = Parallel(n_jobs=jobs, require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor_local.feature_extractor_local)(nlp, entity_recognizer, url, univ_rank, html_dir) for url in urls_compute)
            web_content += [re[0] for re in res]
            abstract_data += [re[1] for re in res]
            print(res)
        except:
            print('go divide')
            for ur in urls_compute:
                try:
                    res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor_local.feature_extractor_local)(nlp, entity_recognizer, url, univ_rank, html_dir) for url in [ur])
                    web_content += [re[0] for re in res]
                    abstract_data += [re[1] for re in res]
                except:
                    web_content.append(['Empty'])
                    abstract_data.append([ur,[]])


    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_content, f)


#feature_dir = './unpaywall_feature_data/'
#html_dir = './unpaywall_middle_result/'
#unpaywall_path = './unpaywall_screenshots/'

feature_dir = "D:\\predatory_journal\\release\\labeled_journal_feature_2025\\"
html_dir = 'D:\\predatory_journal\\release\\labeled_journal_middle_results_2022\\'


for s in range(0,100):
    if span*(s+1)<len(all_urls):
        urls = all_urls[span*s:span*(s+1)]
    else:
        urls = all_urls[span*s:]


    if span2*(s+1)<len(all_urls2):
        urls2 = all_urls2[span2*s:span2*(s+1)]
    else:
        urls2 = all_urls2[span2*s:]



    #print(urls)

    print(len(urls))
    print(len(urls2))
    print('getting content feature')
    get_content_feature_local(urls,'web_content_features_1_'+str(s),30,feature_dir,html_dir)
    print('half way')
    get_content_feature_local(urls2,'web_content_features_0_'+str(s),30,feature_dir,html_dir)
    print('getting tag feature')
    get_tag_feature_local(urls,'web_tag_features_1_'+str(s),feature_dir,html_dir,24)
    print('half way')
    get_tag_feature_local(urls2,'web_tag_features_0_'+str(s),feature_dir,html_dir,24)
    #print('getting font feature')
    #get_font_feature_local(urls,'web_font_features_1_'+str(s),feature_dir,html_dir,24)
    #print('half way')
    #get_font_feature_local(urls2,'web_font_features_0_'+str(s),feature_dir,html_dir,24)



exit()

