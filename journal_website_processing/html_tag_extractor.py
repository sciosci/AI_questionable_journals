from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

import numpy as np
import pandas as pd


def get_html_tag(url,chrome_path):

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    prefs = {
    "download_restrictions": 3,
    "download.default_directory": "D:\\predatory_journal\\release\\null"
     }
    chrome_options.add_experimental_option(
         "prefs", prefs
    )



    try:
        driver = webdriver.Chrome(chrome_path,chrome_options=chrome_options)
        driver.set_page_load_timeout(60)
        driver.get(url)
        html = driver.page_source
        driver.quit()
    except:
        return ''


    soup = BeautifulSoup(html, "html5lib")
    structure = ' '.join([tag.name for tag in soup.find_all()])
    if structure:
        return structure
    else:
        return ''
