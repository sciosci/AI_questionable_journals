# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
get features for each journal
"""


import cv2
import random
import numpy as np
import color_extractor
from sklearn.feature_extraction.text import TfidfVectorizer
import font_extractor
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
import pickle
import shutil
import random

random.seed(5)


df = pd.read_csv('Unpaywall_journal_list_In_MAG_url_all_unique_normalizedname.csv')

screenshot_path = './unpaywall_screenshots'
from pathlib import Path
step = 100

all_urls = df['URL']

random.shuffle(all_urls)



request = ['about','aims','editor','ethics policy','open access policy','copyright policy','latest','paper']
info_key_terms, entity_recognizer, nlp, univ_rank, chrome_options = website_content_extractor.initilize_web_content_extractor()

span = int((len(all_urls)+step-(len(all_urls)%step))/step)

chrome_path = './tools/chromedriver'


def get_content_feature(urls,file_name,jobs,feature_dir,saving_dir,chrome_path):

    web_content = []
    abs_links = []
    for i in range(0,len(urls),jobs):
        urls_compute = urls[i:i+jobs]
        try:
            res = Parallel(n_jobs=jobs, require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in urls_compute)
            web_content += [re[0] for re in res]
            abs_links += [re[1] for re in res]
        except:
            try:
                res = Parallel(n_jobs=int(jobs/2), require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in urls_compute[:int(jobs/2)])
                web_content += [re[0] for re in res]
                abs_links += [re[1] for re in res]
            except:
                print('go divide')
                for ur in urls_compute[:int(jobs/2)]:
                    try:
                        res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in [ur])
                        web_content += [re[0] for re in res]
                        abs_links += [re[1] for re in res]
                    except:
                        web_content.append(['Empty'])
                        abs_links.append([ur,[]])
            try:
                res = Parallel(n_jobs=int(jobs/2), require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in urls_compute[int(jobs/2):])
                web_content += [re[0] for re in res]
                abs_links += [re[1] for re in res]
            except:
                print('go divide')
                for ur in urls_compute[int(jobs/2):]:
                    try:
                        res = Parallel(n_jobs=2, require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path) for url in [ur])
                        web_content += [re[0] for re in res]
                        abs_links += [re[1] for re in res]
                    except:
                        web_content.append(['Empty'])
                        abs_links.append([ur,[]])

    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_content, f)

    abstract_data = []
    for abs in abs_links:
        if len(abs[1]) == 0:
            abstract_data.append([])
        else:
            saving_path = saving_dir + abs[0].replace('/','_')
            try:
                tmp = Parallel(n_jobs=jobs, timeout = 300,verbose=10)(delayed(website_content_extractor.check_abstract)(abs_link,chrome_options,saving_path,chrome_path) for abs_link in abs[1])
                abstract_data.append(tmp)
            except:
                abstract_data.append([])

    with open(feature_dir+file_name+'_abs.pkl', 'wb') as f:
        pickle.dump(abstract_data, f)

def get_tag_feature(urls,file_name,feature_dir):
    web_html_tag_feature = Parallel(n_jobs=12, verbose=10)(delayed(html_tag_extractor.get_html_tag)(url) for url in urls)
    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_html_tag_feature, f)

def get_font_feature(urls,file_name,feature_dir):
    web_font_feature = Parallel(n_jobs=12, verbose=10)(delayed(font_extractor.get_font)(url) for url in urls)
    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(web_font_feature, f)

def get_color_feature(directory,file_name,feature_dir):
    website_colors = []
    color_names, color_rgb = color_extractor.primary_colors('css')
    for im in directory:
        if im != 'None':
            color_class = color_extractor.get_color(im, 5, color_names, color_rgb, 0)
            website_colors.append(color_class)
        else:
            website_colors.append(['None'])

    with open(feature_dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(website_colors, f)

feature_dir = './feature_data/'
saving_dir = './middle_result/'


for s in range(0,step):
    if span*(s+1)<len(all_urls):
        urls = all_urls[span*s:span*(s+1)]
    else:
        urls = all_urls[span*s:]


    print(urls)
    #print(len(urls))
    #print(len(urls2))
    print('getting content feature')
    #urls = ['http://www.dieweltdertuerken.org/'] #['http://www.barentsinfo.org/barentsstudies/English']
    #jobs = 12
    #for url in urls2[-2:]:
    #    web_content1 = website_content_extractor.feature_extractor(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir,chrome_path)
    #    print(web_content1)
    #web_content1 = Parallel(n_jobs=int(jobs/2), require='sharedmem', timeout = 300,verbose=10)(delayed(website_content_extractor.feature_extractor)(nlp, entity_recognizer, url, request, univ_rank, info_key_terms, chrome_options,saving_dir) for url in urls[:int(jobs/2)])
    #urls = ['https://airccse.org/journal/ijsc/ijsc.html']
    get_content_feature(urls,'web_content_features_'+str(s),12,feature_dir,saving_dir,chrome_path)

    print('getting tag feature')
    get_tag_feature(urls,'web_html_tag_features_'+str(s),feature_dir)

    print('getting font feature')
    get_font_feature(urls,'web_font_features_'+str(s),feature_dir)


    screenshot_directory = []

    for url in urls:
        dir = [str(p) for p in Path(screenshot_path).glob("*.png") if url.replace('/','_') in str(p)]
        if len(dir):
            screenshot_directory.append(dir[0])
        else:
            screenshot_directory.append('None')


    print('getting color feature')
    get_color_feature(screenshot_directory,'web_color_features_'+str(s),feature_dir)



exit()

#get_content_feature(urls,'web_content_features_1')
#get_content_feature(urls2,'web_content_features_0')

web_features1 = []
web_features0 = []

with open('web_content_features_1'+'.pkl', 'rb') as f:
    web_features1 = pickle.load(f)

with open('web_content_features_0'+'.pkl', 'rb') as f:
    web_features0 = pickle.load(f)

print('content feature')
print(len(web_features1))
print(len(web_features0))

web_features1 = [ft for ft in web_features1 if len(ft) >9]

unwhite_tb = pd.DataFrame(web_features1)
unwhite_tb['label'] = 1

web_features0 = [ft for ft in web_features0 if len(ft) >9]
white_tb = pd.DataFrame(web_features0)
white_tb['label'] = 0

all_tb = pd.concat([unwhite_tb,white_tb]).reset_index(drop=True)

X = all_tb.loc[:, all_tb.columns != 'label']
y = all_tb.loc[:, all_tb.columns == 'label']


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(max_depth=5, random_state=0)
scores = cross_val_score(clf, X, y, cv=2)

print(scores)


#exit()




#get_tag_feature(urls,'web_html_tag_features_1')
#get_tag_feature(urls2,'web_html_tag_features_0')


tag_features1 = []
tag_features0 = []

with open('web_html_tag_features_1'+'.pkl', 'rb') as f:
    tag_features1 = pickle.load(f)

with open('web_html_tag_features_0'+'.pkl', 'rb') as f:
    tag_features0 = pickle.load(f)

print('tag feature')
print(len(tag_features1))
print(len(tag_features0))

data = []

[data.append(ft) for ft in tag_features1 if len(ft)>=1]
[data.append(ft) for ft in tag_features0 if len(ft)>=1]
label = [1 for ft in tag_features1 if len(ft)>=1] + [0 for ft in tag_features0 if len(ft)>=1]



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
y = np.array(label)


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(max_depth=5, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)


print(scores)


def get_font_feature(urls,file_name):
    web_font_feature = Parallel(n_jobs=12)(delayed(font_extractor.get_font)(url) for url in urls)
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(web_font_feature, f)

get_font_feature(urls,'web_font_features_1')
get_font_feature(urls2,'web_font_features_0')


with open('web_font_features_1'+'.pkl', 'rb') as f:
    font_features1 = pickle.load(f)

with open('web_font_features_0'+'.pkl', 'rb') as f:
    font_features0 = pickle.load(f)

print('font feature')
print(len(font_features1))
print(len(font_features0))

font_feature = [ft for ft in font_features1 if len(ft)>0] + [ft for ft in font_features0 if len(ft)>0]
label = [1 for ft in font_features1 if len(ft)>0] + [0 for ft in font_features0 if len(ft)>0]

print(len(font_feature))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([' '.join(ff) for ff in font_feature])
y = np.array(label)


cv = StratifiedKFold(n_splits=5,random_state = 231,shuffle=True)
classifier = svm.SVC( random_state=5)
#classifier = LogisticRegression(random_state=5)
#classifier = LogisticRegressionCV(random_state=5)
#classifier = RandomForestClassifier(max_depth=5,n_estimators = 200, random_state=0)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic example",
)
ax.legend(loc="lower right")
plt.show()

print(scores)


exit()


unwhite_path = '/home/hzhuang/Projects/Predatory_Journal/predatory_journal/predatory_journals_clean/screenshots_model/unwhite/'
white_path = '/home/hzhuang/Projects/Predatory_Journal/predatory_journal/predatory_journals_clean/screenshots_model/white/'
from pathlib import Path

unwhite_directory = []

for url in urls:
    dir = [str(p) for p in Path(unwhite_path).glob("*.png") if url.replace('/','_') in str(p)]
    if len(dir):
        unwhite_directory.append(dir[0])
    else:
        unwhite_directory.append('None')



white_directory = []

for url in urls2:
    dir = [str(p) for p in Path(white_path).glob("*.png") if url.replace('/','_') in str(p)]
    if len(dir):
        white_directory.append(dir[0])
    else:
        white_directory.append('None')

#[str(p) for p in Path(unwhite_path).glob("*.png") if len([url for url in urls if url.replace('/','_') in str(p)])]
print(len(unwhite_directory))
#white_directory = [str(p) for p in Path(white_path).glob("*.png") if len([url for url in urls2 if url.replace('/','_') in str(p)])]
print(len(white_directory))

color_names, color_rgb = color_extractor.primary_colors('css')
# Press the green button in the gutter to run the script.

website_colors = []
label = []

def get_color_feature(directory,file_name):
    website_colors = []
    for im in directory:
        if im != 'None':
            color_class = color_extractor.get_color(im, 5, color_names, color_rgb, 0)
            website_colors.append(color_class)
        else:
            website_colors.append(['None'])

    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(website_colors, f)

get_color_feature(unwhite_directory,'web_color_features_1')
get_color_feature(white_directory,'web_color_features_0')

with open('web_color_features_1'+'.pkl', 'rb') as f:
    color_features1 = pickle.load(f)

with open('web_color_features_0'+'.pkl', 'rb') as f:
    color_features0 = pickle.load(f)

print('color feature')
print(len(color_features1))
print(len(color_features0))

website_colors = [ft for ft in color_features1 if len(ft)>0] + [ft for ft in color_features0 if len(ft)>0]
label = [1 for ft in color_features1 if len(ft)>0] + [0 for ft in color_features0 if len(ft)>0]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([' '.join(set(ff)) for ff in website_colors])
y = np.array(label)



from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(max_depth=5, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)

print(scores)



exit()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
y = np.array(y)


cv = StratifiedKFold(n_splits=3,random_state = 11131,shuffle=True)
classifier = RandomForestClassifier(max_depth=5, random_state=0)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    print(np.mean(tprs, axis=0))

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print(mean_auc)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic example",
)
ax.legend(loc="lower right")
plt.show()









exit()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
