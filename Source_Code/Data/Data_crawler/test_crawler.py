import datetime
import pytz
import os, re
import requests

import urllib3
from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.action_chains import ActionChains

chrome_options = Options()
chrome_options.add_argument("--incognito")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("user-agent=Chrome/80.0.3987.132")
# chrome_options.add_argument("--headless")

# //------------------------------------Test crawl song urls-----------------------------------------------//

driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=r"E:\Data_analysis\Data_Science\Data_crawler\chromedriver.exe")
driver.get('https://open.spotify.com/playlist/37i9dQZF1DWWH0izG4erma')
sleep(2)

try:
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()
    print("accepted cookies")
except Exception as e:
    print('no cookie button')


bottom_sentinel = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@data-testid='bottom-sentinel']")))
song_arr = []
reached_page_end = False

f = open(r'E:\Data_analysis\Data_Science\test.txt', 'w', encoding='utf-8')
while not reached_page_end:
    sleep(1)

    table = BeautifulSoup(driver.page_source, 'html.parser')
    song_list = table.findAll("div", {'data-testid': 'tracklist-row'})
    test = 0
    for song in song_list:
        song_info = song.find('a', {'class': 't_yrXoUO3qGsJS4Y6iXX'})
        song_link = 'https://open.spotify.com' + str(song_info.get('href'))
        song_name = song_info.find('div', {'class': 'Type__TypeElement-sc-goli3j-0 kHHFyx t_yrXoUO3qGsJS4Y6iXX standalone-ellipsis-one-line'})
        sn = ''
        if len(song_name.contents) != 0 :
            sn = song_name.contents[0]

        if song_link not in song_arr:
            test += 1
            song_arr.append(str(song_link))
            f.write(str(song_link + ' ' + sn + '\n'))
            
    bottom_sentinel.location_once_scrolled_into_view
    driver.implicitly_wait(15)

    if test == 0:
        reached_page_end = True
    else:
        print('pass')

# //------------------------------------Test crawl song lyric-----------------------------------------------//

driver = webdriver.Chrome(chrome_options=chrome_options, 
        executable_path=r"E:\Data_analysis\Data_Science\Data_crawler\chromedriver.exe")

driver.get(r'https://accounts.spotify.com/en/login?error=errorFacebookAccount&continue=https%3A%2F%2Fwww.spotify.com%2Fus%2Faccount%2Flogin%2F')
sleep(2)
driver.find_element(By.ID, "login-username").send_keys("quyettzzlczz@gmail.com")
sleep(1)
driver.find_element(By.ID, "login-password").send_keys("12345678lc")
sleep(1) 
driver.find_element(By.XPATH, '//*[@id="login-button"]/div[1]').click()
sleep(10)

driver.get('https://open.spotify.com/track/3CqtQwaoQTdgDWLHyv7Twr')
sleep(5)
soup = BeautifulSoup(driver.page_source.encode('utf-8').strip(), 'html.parser')
lyric_table = soup.find('div', {'class': "Q3OKWaFrTVTIRZyG05Gh"}).findAll('p', {'class': 'Type__TypeElement-goli3j-0 ipKmGr NAaJboGa8qckhNNQTKTn'})
sleep(5)

for sen in lyric_table:
    print(sen.text)

# print(driver.page_source)

