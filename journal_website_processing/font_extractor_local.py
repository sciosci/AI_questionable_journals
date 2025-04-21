from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlparse
import cssutils
import requests
import logging
import os

"""
get fonts for each image
"""

def get_font(saving_dir,url):

    fonts = []
    cssutils.log.setLevel(logging.CRITICAL)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    prefs = {
    "download_restrictions": 3,
    "download.default_directory": "./null/"
     }
    chrome_options.add_experimental_option(
         "prefs", prefs
    )

    try:
        driver = webdriver.Chrome('/home/hzhuang/.wdm/drivers/chromedriver/linux64/96/chromedriver',chrome_options=chrome_options)
        driver.set_page_load_timeout(60)
        driver.get(url)
        driver.get(driver.current_url)
    except:
        return fonts

    host_url = urlparse(driver.current_url).hostname
    html = driver.page_source

    soup = BeautifulSoup(html, "html5lib")


    driver.quit()

    saving_path = saving_dir + url.replace('/','_') + "/css_files/"

    if not os.path.exists(saving_path):
        os.mkdir(saving_path)


    all_style = ''

    css_links = []
    for styletag in soup.findAll('style'):
        if styletag.string:
            all_style += styletag.string


    for styletag in soup.findAll('link',type="text/css"):

        if styletag.has_attr('href'):
            css_herf = styletag['href']
        else:
            css_herf = ''

        if 'http' not in css_herf and css_herf not in css_links and len(css_herf)>0:
            css_links.append(css_herf)
            css_url = 'http:' + css_herf
            try:
                html_text = requests.get(css_url, timeout=30).text
                with open(saving_path+css_url, "w") as file:
                    file.write(str(html_text))
            except:
                if 'http' not in host_url:
                    if len(host_url) > 4 and host_url[-1] != '/' and css_herf[0] != '/':
                        try:
                            html_text = requests.get('http://'+host_url+'/'+css_herf, timeout=30).text
                            with open(saving_path+css_herf, "w") as file:
                                file.write(str(html_text))
                        except:
                            pass
                    else:
                        pass
                else:
                    if len(host_url) > 4 and host_url[-1] != '/' and css_herf[0] != '/':
                        try:
                            html_text = requests.get(host_url+'/'+css_herf, timeout=30).text
                            with open(saving_path+css_herf, "w") as file:
                                file.write(str(html_text))
                        except:
                            pass
                    else:
                        pass


        elif css_herf not in css_links:
            css_links.append(css_herf)
            css_url = css_herf
            try:
                html_text = requests.get(css_url, timeout=30).text
                with open(saving_path+css_url, "w") as file:
                    file.write(str(html_text))
            except:
                continue

        all_style += html_text

    try:
        sheet = cssutils.parseString(all_style)
    except:
        return fonts



    for rule in sheet:
        if rule.type == rule.STYLE_RULE:
            #find property
            for property in rule.style:
                if property.name == 'font-family':
                    fonts.append(property.value)

    return fonts

def get_font_local(html_dir,url):
    # used the data from windows system
    fonts = []
    cssutils.log.setLevel(logging.CRITICAL)

    home_path = html_dir + url.replace('/','_')
    if not os.path.isdir(home_path):
        print('not exist')
        return fonts

    with open(home_path+"/home.html", "r") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html5lib")

    all_style = ''

    for styletag in soup.findAll('style'):
        if styletag.string:
            all_style += styletag.string

    saved_path = html_dir + url.replace('/','_') + "/css_files/"
    #saved_path = html_dir + url.replace('/','_').replace(':','-').replace('?','~') + "/css_files/"

    if os.path.isdir(saved_path):

        for css_files in os.listdir(saved_path):
            with open(saved_path+css_files, "r",encoding='windows-1252') as f:
                html_text = f.read()

            all_style += html_text



    try:
        sheet = cssutils.parseString(all_style)
    except:
        return fonts



    for rule in sheet:
        if rule.type == rule.STYLE_RULE:
            #find property
            for property in rule.style:
                if property.name == 'font-family':
                    fonts.append(property.value)

    return fonts
