from bs4 import BeautifulSoup

import numpy as np
import pandas as pd


def get_html_tag_local(html_dir,url):

    saving_path = html_dir + url.replace('/','_')

    with open(saving_path+"/home.html", "r") as f:
        html = f.read()


    soup = BeautifulSoup(html, "html5lib")
    structure = ' '.join([tag.name for tag in soup.find_all()])
    if structure:
        return structure
    else:
        return ''
