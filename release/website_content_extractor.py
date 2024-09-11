import spacy
import pandas as pd
import numpy as np
import random, os
from bs4 import BeautifulSoup
from selenium import webdriver
import pickle
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import requests
import textstat
from urllib.request import urlopen
from urllib.parse import urlparse

def get_lang_detector(nlp, name):
    return LanguageDetector()

def initilize_web_content_extractor():
    info_key_terms = {'about': ['about'],
                  'aims': ['aims & scope','aims and scope','focus and scope','aim and scope','aims and objectives','scope','overview'],
                  'editor': ['view editorial board','editorial team','editorial board','editors','journal boards','editorial'],
                  'ethics policy': ['statement of ethics','ethical considerations',
                             'publication ethics and malpractice statement','ethics', 'ethic',
                             'editorial policies','editorial policy','author guidelines',
                             'submission guidelines','instructions for authors','instructions to authors',
                             'guide for authors','submit an article','publication ethics','for authors',
                             'policies','guidelines','authors'],
                  'open access policy': ['article processing charges','open access'],
                  'copyright policy': ['license','licence','copyright'],
                  'latest': ['lastest','current','recent'],
                  'paper': ['archive','issue','publications','articles'],
                  'volume': ['vol', 'no','issue'],
                  'abstract': ['abs', 'abstract','full text']
                 }

    spacy.prefer_gpu()
    entity_recognizer = spacy.load("en_core_web_sm")
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    univ_rank = pd.read_csv('Affiliation_rank.csv')
    univ_rank['institution'] = univ_rank['NormalizedName']


    chrome_options = Options()
    chrome_options.add_argument("--headless")
    prefs = {
    "download_restrictions": 3,
    "download.default_directory": "./null/"
     }
    chrome_options.add_experimental_option(
         "prefs", prefs
    )

    return info_key_terms, entity_recognizer, nlp, univ_rank, chrome_options


def recoginize_institution(entity_recognizer,univ_rank,text):

    doc = entity_recognizer(text)
    institues = []
    rank = []
    for ent in doc.ents:

        if ent.label_ =='ORG':

            if 'university' in ent.text.lower() or 'college' in ent.text.lower() or 'institute' in ent.text.lower():

                tmp = univ_rank[univ_rank['institution'] == ent.text.lower()]
                institues.append(ent.text.lower())
                if tmp.shape[0]>0:
                    rank.append(tmp['Rank'].iloc[0])


    if len(rank) >=2:
        score = np.mean(rank)
    else:
        score = max(univ_rank['Rank'])

    return [score, len(institues)]

def analyze_editorial_board(url,entity_recognizer,univ_rank,chrome_options,saving_path,chrome_path):

    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")

    with open(saving_path+"/editors.html", "w") as file:
        file.write(str(soup))

    content_1 = []
    content_2 = []
    content_3 = []

    for tag in soup.findAll("p"):
        content_1 += [item.strip() for item in tag.getText(separator=' ').split('\n')]

    for tag in soup.findAll("div"):
        content_2 += [item.strip() for item in tag.getText(separator=' ').split('\n')]

    for tag in soup.findAll("br"):
        content_3 += [item.strip() for item in tag.getText(separator=' ').split('\n')]

    content = list(set(content_1))+list(set(content_2))+list(set(content_3))

    driver.quit()
    return recoginize_institution(entity_recognizer,univ_rank," ".join(content))

def analyze_aims(url,chrome_options,saving_path,chrome_path):

    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")

    with open(saving_path+"/aims.html", "w") as file:
        file.write(str(soup))

    content = [tag.getText(separator=' ').split('\n')[0] for tag in soup.findAll("p")] + [tag.getText(separator=' ').split('\n')[0] for tag in soup.findAll("div")]
    aims = max(content, key=len)
    driver.quit()
    results = [textstat.flesch_reading_ease(aims),
               textstat.flesch_kincaid_grade(aims),
               textstat.smog_index(aims),
               textstat.coleman_liau_index(aims),
               textstat.dale_chall_readability_score(aims),
               textstat.automated_readability_index(aims),
               textstat.difficult_words(aims),
               textstat.linsear_write_formula(aims),
               textstat.gunning_fog(aims),
               textstat.reading_time(aims, ms_per_char=14.69)]
    return results

def check_abstract(url,chrome_options,saving_dir,chrome_path):


    keys = {'abstract':['abs', 'abstract','full text']}
    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")


    saving_path = saving_dir +'/latest_issue_paper/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)


    with open(saving_path+url.replace('/','_'), "w") as file:
        file.write(str(soup))



    content = []
    #limited to div and p to speed up?
    for tag in soup.findAll():

        try:
            if tag.has_attr('id') and link_class(tag['id'], keys)[1] in keys:
                content.append(tag.getText(separator=' '))
            elif link_class(tag.text, keys)[1] in keys:
                content.append(tag.getText(separator=' '))
            elif tag.has_attr('class') and link_class(' '.join(tag['class']), keys)[1] in keys:
                content.append(tag.getText(separator=' '))
        except:
            pass

    if len(content) == 0:
        return []
    abstract = max(content, key=len)

    driver.quit()
    results = [textstat.flesch_reading_ease(abstract),
               textstat.flesch_kincaid_grade(abstract),
               textstat.smog_index(abstract),
               textstat.coleman_liau_index(abstract),
               textstat.dale_chall_readability_score(abstract),
               textstat.automated_readability_index(abstract),
               textstat.difficult_words(abstract),
               textstat.linsear_write_formula(abstract),
               textstat.gunning_fog(abstract),
               textstat.reading_time(abstract, ms_per_char=14.69)]
    return results

def analyze_issues(url,chrome_options,saving_path,chrome_path):
    #'pdf', 'html', 'full text',
    keys = {'paper': ['vol','volume','issue']}

    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")

    with open(saving_path+"/archive.html", "w") as file:
        file.write(str(soup))

    found_paper = []
    for tag in soup.findAll("a"):
        if link_class(tag.text, keys)[1] in keys:
            found_paper.append(1)

    driver.quit()
    if len(found_paper)>0:
        found = 1
    else:
        found = 0
    return found


def analyze_policy(url,chrome_options,saving_path,chrome_path):
    keys = {'authorship': ['authorship'],
            'conflict': ['conflict of interest', 'conflicts of interest', 'competing interests', 'conflicting interests'],
            'copyright and license': ['copyright', 'license'],
            'peer review': ['single-blind', 'double-blind', 'single blind', 'double blind','peer review']
           }


    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")

    saving_path = saving_path +'/policy/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    with open(saving_path+url.replace('/','_'), "w") as file:
        file.write(str(soup))

    found_policy = []
    for tag in soup.findAll():
        link_type = link_class(tag.text, keys)[1]
        if link_type in keys:
            found_policy.append(link_type)

    all_text = soup.getText(separator=' ').lower()

    for content_type, titles in keys.items():
        for ind, title in enumerate(titles):
            if title in all_text:
                if content_type not in found_policy:
                    found_policy.append(content_type)

    driver.quit()

    result = []
    for c, t in keys.items():
        if c in found_policy:
            result.append(1)
        else:
            result.append(0)
    return result

def link_class(text: str, keys: dict):

    text = text.strip()
    for content_type, titles in keys.items():
        for ind, title in enumerate(titles):
            if title in text.lower() and len(text.lower())<=30:

                ind = titles.index(title.lower())

                return [ind, content_type]
    return [-1, 'empty']


def check_link(all_tags, i, info_key_terms,requested_info, host_url):
    tag = all_tags[i]
    found_info = dict()
    link = tag.get('href', None)

    if link:

        if ('/articles/' in link or '/article/' in link or '/doi/' in link) and requested_info =='abstract':

            try:
                if 'http' not in link:
                    new_link = 'http://' + link
                else:
                    new_link = link

                requests.get(new_link)

                found_info[requested_info] = [0,new_link]
            except:

                if host_url[-1] != '/' and link[0] != '/':
                    host_url += '/'
                if host_url[-1] == '/' and link[0] == '/':
                    link = link[1:]

                found_info[requested_info] = [0,host_url+link]

            #print('checked')
            return found_info
        #print({requested_info: info_key_terms[requested_info]})
        #print(tag.text)
        ind, text = link_class(tag.text,{requested_info: info_key_terms[requested_info]})
        #print("class:"+text)
        if text in requested_info and text != 'empty':

            try:
                if 'http' not in link:
                    new_link = 'http://' + link
                else:
                    new_link = link

                requests.get(new_link)

                found_info[requested_info] = [ind,new_link]
            except:

                if host_url[-1] != '/' and link[0] != '/':
                    host_url += '/'
                if host_url[-1] == '/' and link[0] == '/':
                    link = link[1:]

                found_info[requested_info] = [ind,host_url+link]


    return found_info

def analyze_latest_issue(nlp, url,request,info_key_terms,chrome_options,saving_path,chrome_path):

    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")

    with open(saving_path+"/latest_issue.html", "w") as file:
        file.write(str(soup))

    res4 = retrieve_href(nlp,url,request,info_key_terms,chrome_options,chrome_path)
    if 'abstract' in res4:
        return res4['abstract']
    else:
        return []


def retrieve_href(nlp,url: str,requested_info: list, info_key_terms,chrome_options,chrome_path) -> dict:

    found_info = {}
    all_info = []
    driver = webdriver.Chrome(chrome_path,options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.get(url)
    driver.get(driver.current_url)


    root_url = urlparse(driver.current_url).hostname

    if 'http' not in root_url:
        root_url = 'http://' + root_url

    html = driver.page_source

    soup = BeautifulSoup(html,features="lxml")

    doc = nlp(soup.text)

    if doc._.language['language'] != 'en':
        driver.quit()
        return {}


    all_info = []

    all_tags = soup.find_all("a")


    for request in requested_info:
        #print(request)
        for i in range(len(all_tags)):
            #print('tag:'+str(all_tags[i]))
            tp = check_link(all_tags,i,info_key_terms,request,root_url)
            #print('type:'+str(tp))
            all_info.append(tp)
            if request == 'volume' and len(tp)>0 and list(tp.values())[0][0] ==0:
                break


    all_info = [info for info in all_info if len(info) > 0]

    if requested_info[0] =='abstract':
        return {'abstract':[info['abstract'][1] for info in all_info]}

    for info in all_info:

        if list(info.keys())[0] not in list(found_info.keys()):
            found_info[list(info.keys())[0]] = list(info.values())[0]
        elif list(info.values())[0][0] < found_info[list(info.keys())[0]][0]:
            found_info[list(info.keys())[0]] = list(info.values())[0]


    driver.quit()

    return found_info

def feature_extractor(nlp,entity_recognizer,url,request,univ_rank,info_key_terms,chrome_options,saving_dir,chrome_path):


    res = None
    aim_feature = [0]*10
    editor_feature = [max(univ_rank['Rank']),0]
    policy_feature = [-1,-1,-1,-1]
    all_abs = []
    res2 = None


    #print(url)

    driver = webdriver.Chrome(chrome_path,options=chrome_options)

    driver.get(url)
    driver.get(driver.current_url)
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")

    driver.quit()

    saving_path = saving_dir + url.replace('/','_')

    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    with open(saving_path+"/home.html", "w") as file:
        file.write(str(soup))


    try:
        res = retrieve_href(nlp,url,request,info_key_terms,chrome_options,chrome_path)
    except:
        result = ['Empty'] + editor_feature+ ['Empty'] + policy_feature
        return [result,[url,[]]]


    if res == {}:
        result = ['Non-EN'] + editor_feature+ ['Non-EN'] + policy_feature
        return [result,[url,[]]]

    if res and 'about' in res.keys():
        if 'editor' not in res.keys() or 'aims' not in res.keys() or 'ethics policy' not in res.keys():
            try:
                res2 = retrieve_href(nlp,res['about'][1],request,info_key_terms,chrome_options,chrome_path)
                #print('2')
            except:
                pass
            if res2 and 'editor' in res2.keys():
                res['editor'] = res2['editor']

        if 'aims' not in res.keys():
            if res2 and 'aims' in res2.keys():
                res['aims'] = res2['aims']

        if 'ethics policy' not in res.keys():
            if res2 and 'ethics policy' in res2.keys():
                res['ethics policy'] = res2['ethics policy']

        if 'open access policy' not in res.keys():
            if res2 and 'open access policy' in res2.keys():
                res['open access policy'] = res2['open access policy']

        if 'copyright policy' not in res.keys():
            if res2 and 'copyright policy' in res2.keys():
                res['copyright policy'] = res2['copyright policy']


    #print(res)
    if res and 'aims' in res.keys():
        try:
            aim_feature = analyze_aims(res['aims'][1],chrome_options,saving_path,chrome_path)
            #print('3')
        except:
            pass

    if res and 'editor' in res.keys():
        try:
            editor_feature = analyze_editorial_board(res['editor'][1],entity_recognizer,univ_rank,chrome_options,saving_path,chrome_path)
            #print('4')
        except:
            pass

    if res and 'ethics policy' in res.keys():
        try:
            policy_feature = analyze_policy(res['ethics policy'][1],chrome_options,saving_path,chrome_path)

        except:
            pass

    if res and 'open access policy' in res.keys():
        try:
            tmp = analyze_policy(res['open access policy'][1],chrome_options,saving_path,chrome_path)

            for ind, pf in enumerate(tmp):
                if pf == 1 and policy_feature[ind] <= 0:
                    policy_feature[ind] = 1

        except:
            pass

    if res and 'copyright policy' in res.keys():
        try:
            tmp = analyze_policy(res['copyright policy'][1],chrome_options,saving_path,chrome_path)

            for ind, pf in enumerate(tmp):
                if pf == 1 and policy_feature[ind] <= 0:
                    policy_feature[ind] = 1

        except:
            pass

    if res and 'latest' in res.keys():

        all_abs = analyze_latest_issue(nlp, res['latest'][1], ['abstract'],info_key_terms,chrome_options,saving_path,chrome_path)


    if res and 'latest' not in res.keys() and 'paper' in res.keys():
        try:
            res3 = retrieve_href(nlp,res['paper'][1],['volume'],info_key_terms,chrome_options,chrome_path)
            #print('7')
            #print(url)

            res4 = retrieve_href(nlp,res3['volume'][1],['abstract'],info_key_terms,chrome_options,chrome_path)
            #print('8')
            all_abs = res4['abstract']

        except:
            pass

    with open(saving_path+'/'+'key_pages.pkl', 'wb') as f:
        pickle.dump(res, f)

    with open(saving_path+'/'+'latest_paper_pages.pkl', 'wb') as f:
        pickle.dump(all_abs, f)

    result = aim_feature + editor_feature + policy_feature
    return [result, [url,all_abs]]
