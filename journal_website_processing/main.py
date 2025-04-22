# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
get features for each journal
"""


import cv2
import random
import numpy as np
#import color_extractor
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
import website_content_extractor
import html_tag_extractor
from joblib import Parallel, delayed
import pickle,os
import shutil
import random



#df = pd.read_csv('Unpaywall_journal_list_In_MAG_url_all_unique_normalizedname.csv')
#df = pd.read_csv('missed_unpaywall_journals_to_check_and_add_to_unpaywall_middle_result.csv')
#df = pd.read_csv('missing_labeled_website_folder_to_check_and_add_if_need.csv')
#df = pd.read_csv('doaj_journal_list_urls_from_doaj_bing.csv').sample(frac=1,random_state =6)
#df.to_csv('journals_scraped.csv',index=False)
from pathlib import Path

df = pd.read_csv('..\\data\\doaj_journal_list_urls_from_doaj_bing_v1.csv')
#df = pd.read_csv("..\\survey_analysis\\Journals Survey No. 1 - Journals.csv")
#resume previous downloading
#df = df[~df['ISSN'].isin(os.listdir('..\\labeled_journal_middle_results_2022\\'))]
step = 10

all_urls = df['Website'].tolist()
all_journals = df['NormalizedName'].tolist()

print(len(all_urls))
#random.shuffle(all_urls)


#I removed to requests 'latest','paper'
request = ['about','aims','editor','ethics policy','open access policy','copyright policy']
info_key_terms, entity_recognizer, nlp, univ_rank, chrome_options = website_content_extractor.initilize_web_content_extractor()

span = int((len(all_urls)+step-(len(all_urls)%step))/step)

chrome_path = 'chromedriver.exe'


def download_websites(urls, journals, jobs, saving_dir, chrome_path):

    web_content = []
    abs_links = []
    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]
        journals_compute = journals[i:i+jobs]
        try:
            res = Parallel(n_jobs=jobs, require='sharedmem', timeout = 180,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, journals_compute[ind], request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for ind,url in enumerate(urls_compute))
            #web_content += [re[0] for re in res]
            #abs_links += [re[1] for re in res]
        except:
            try:
                res = Parallel(n_jobs=int(jobs/2), require='sharedmem', timeout = 180,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, journals_compute[ind], request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for ind,url in enumerate(urls_compute[:int(jobs/2)]))
                #web_content += [re[0] for re in res]
                #abs_links += [re[1] for re in res]
            except:
                print('go divide')
                for ind,ur in enumerate(urls_compute[:int(jobs/2)]):
                    try:
                        res = Parallel(n_jobs=2, require='sharedmem', timeout = 120,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, journals_compute[ind], request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in [ur])
                        #web_content += [re[0] for re in res]
                        #abs_links += [re[1] for re in res]
                    except:
                        web_content.append(['Empty'])
                        abs_links.append([ur,[]])
            try:
                res = Parallel(n_jobs=int(jobs/2), require='sharedmem', timeout = 180,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, journals_compute[ind], request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for ind,url in enumerate(urls_compute[int(jobs/2):]))
                #web_content += [re[0] for re in res]
                #abs_links += [re[1] for re in res]
            except:
                print('go divide')
                for ind,ur in enumerate(urls_compute[int(jobs/2):]):
                    try:
                        res = Parallel(n_jobs=2, require='sharedmem', timeout = 120,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, journals_compute[ind], request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in [ur])
                        #web_content += [re[0] for re in res]
                        #abs_links += [re[1] for re in res]
                    except:
                        web_content.append(['Empty'])
                        abs_links.append([ur,[]])




    return

    abstract_data = []
    for abs in abs_links:
        if len(abs[1]) == 0:
            abstract_data.append([])
        else:
            saving_path = saving_dir + abs[0].replace('/','_').replace(':','-')
            try:
                tmp = Parallel(n_jobs=4, timeout = 300,verbose=10)(delayed(website_content_extractor.check_abstract)(abs_link,chrome_options,saving_path,chrome_path) for abs_link in abs[1])
                print(tmp)
                abstract_data.append(tmp)
            except:
                abstract_data.append([])

    with open(feature_dir+file_name+'_abs.pkl', 'wb') as f:
        pickle.dump(abstract_data, f)




def get_tag_feature(urls,file_name,jobs,feature_dir,chrome_path):
    #web_html_tag_feature = Parallel(n_jobs=6, verbose=10)(delayed(html_tag_extractor.get_html_tag)(url,chrome_path) for url in urls)
    #with open(feature_dir+file_name+'.pkl', 'wb') as f:
    #    pickle.dump(web_html_tag_feature, f)
    web_html_tag_feature = []
    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]
        try:
            res = Parallel(n_jobs=jobs, require='sharedmem', timeout = 300,verbose=10)(delayed(html_tag_extractor.get_html_tag)(url,chrome_path) for url in urls_compute)
            web_html_tag_feature += res
        except:
            for ur in urls_compute:
                try:
                    res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(html_tag_extractor.get_html_tag)(url,chrome_path) for url in urls_compute)
                    web_html_tag_feature += res
                except:
                    pass

    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_html_tag_feature, f)

def download_css(urls,journals,jobs,saving_dir,chrome_path):

    web_font_feature = []

    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]
        journals_compute = journals[i:i+jobs]
        try:
            res = Parallel(n_jobs=jobs, timeout = 300, verbose=10)(delayed(font_extractor_local.get_font)(saving_dir,url,journals_compute[ind],chrome_path) for ind,url in enumerate(urls_compute))
            print(res)
            web_font_feature += res
        except:
            for ind,ur in enumerate(urls_compute):
                try:
                    res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(font_extractor_local.get_font)(saving_dir,url,journals_compute[ind],chrome_path) for url in [ur])
                except:
                    continue







#feature_dir = '..\\labeled_journal_feature_data_missed_before\\'
#saving_dir = '..\\labeled_journal_middle_result_missed_before_with_css\\'
#feature_dir = '..\\survey_analysis\\survey_journal_features\\'
#saving_dir = '..\\survey_analysis\\survey_journal_middle_results\\'

print(len(all_urls) == len(all_journals))

for s in range(0,step):
    if span*(s+1)<len(all_urls):
        urls = all_urls[span*s:span*(s+1)]
    else:
        urls = all_urls[span*s:]

    if span*(s+1)<len(all_urls):
        journals = all_journals[span*s:span*(s+1)]
    else:
        journals = all_journals[span*s:]


    #print(len(urls))
    #print(len(urls2))
    print('download websites: '+str(s))
    #urls = ['http://www.dieweltdertuerken.org/'] #['http://www.barentsinfo.org/barentsstudies/English']
    #jobs = 12
    #for url in urls[10:20]:
    #    web_content1 = website_content_extractor.feature_extractor(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path)
    #    print(web_content1)
        #saving_path = saving_dir + web_content1[1][0].replace('/','_').replace(':','-')
        #for abs_link in web_content1[1][1]:
        #    print(website_content_extractor.check_abstract(abs_link,chrome_options,saving_path,chrome_path))
    #web_content1 = Parallel(n_jobs=int(jobs/2), require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir) for url in urls[:int(jobs/2)])
    #urls = ['https://airccse.org/journal/ijsc/ijsc.html']
    download_websites(urls,journals,24,saving_dir,chrome_path)
    #download_css(urls,journals,24,saving_dir,chrome_path)

    #print('getting tag feature')
    #get_tag_feature(urls,'web_html_tag_features_'+str(s),6,feature_dir,chrome_path)

    #print('getting font feature')
    #get_font_feature(urls,'web_font_features_'+str(s),6,feature_dir,saving_dir,chrome_path)

    #print('getting color feature')
    #get_color_feature(screenshot_directory,'web_color_features_'+str(s),feature_dir)



exit()

