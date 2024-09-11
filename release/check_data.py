import numpy as np
import spacy
import pandas as pd

def recoginize_institution(entity_recognizer,univ_rank,text):

    doc = entity_recognizer(text)
    institues = []
    rank = []
    print(doc)
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
    print(score)
    return [score, len(institues)]

from bs4 import BeautifulSoup
import pickle

def analyze_editorial_board(entity_recognizer,univ_rank):
    html = ''
    with open("editors.html", "r", encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html,features="lxml")


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

    return recoginize_institution(entity_recognizer,univ_rank," ".join(content))

spacy.prefer_gpu()
entity_recognizer = spacy.load("en_core_web_sm")
univ_rank = pd.read_csv('Affiliation_rank.csv')
univ_rank['institution'] = univ_rank['NormalizedName']


#analyze_editorial_board(entity_recognizer,univ_rank)
def get_html_tag():

    with open("home.html", "r") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html5lib")
    structure = ' '.join([tag.name for tag in soup.find_all()])

    with open('html_tag_remove.pkl', 'wb') as f:
        pickle.dump(structure, f)

    if structure:
        return structure
    else:
        return ''


get_html_tag()
