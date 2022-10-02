
from googlesearch import search
from newspaper import Article
from newsfetch.news import newspaper
import numpy as np
import re
import string
import jsonlines
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
import pandas as pd
import jsonlines


API_KEY = "AIzaSyAC2fKvrczedZhBfZmSmYDRaYvSF_nm5HU"
SEID = "017236688157949926015:-2mk2err3ps"


# Load all the facts
all_facts = []
with jsonlines.open("/home/zsarwar/NLP/autoprompt/data/Roberta_100_Short.jsonl", 'r') as in_file:
    for fact in in_file:
        af = fact['Pre_Mask'] + fact['Label'] + fact['Post_Mask'].replace(" .", ".")
        all_facts.append(af)



all_entities = []
all_facts = []
with jsonlines.open("/home/zsarwar/NLP/autoprompt/data/Roberta_100_Short.jsonl", 'r') as in_file:
    for fact in in_file:
        curr_fact = fact['Pre_Mask'] + fact['Label'] + fact['Post_Mask'].replace(" .", ".")
        prompt = fact['Prompt'].replace('[Y]', fact['Label']).replace(" .", ".")
        entity = curr_fact
        if prompt[-1] == ".":
            prompt = prompt[0:-1]

        if entity[-1] == ".":
            entity = entity[0:-1]
        
        pre_x = prompt.find('[X]')
        entity = entity[pre_x:]
        post_x_txt = prompt[pre_x+3:]
        if(len(post_x_txt) > 1):
            post_x_idx = entity.find(post_x_txt)
            entity = entity[0:post_x_idx]
        
        all_entities.append(entity)
        all_facts.append(curr_fact)



def google_search_api(search_term, api_key, cse_id, num_results,num_iters,
                       **kwargs):
    start_index = -9
    results = []
    for i in range(num_iters):
        if i == (num_iters - 1):
            if num_iters == 10:
                num_results = (num_results - 1)
        start_index += 10
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id,num = num_results,start = start_index , **kwargs).execute()
        results.append(res)
    return results
def get_urls_api(search_results):
    extracted_urls = set()
    for api_results in search_results:
        for page in api_results:
            if (page.get('items') is not None):
                for i in range(len(page['items'])):
                    ex = page['items'][i]['link']
                    extracted_urls.add(ex)
    return extracted_urls


for idx, fact in enumerate(all_facts[0:5]):
  out_file = jsonlines.open("/home/zsarwar/NLP/autoprompt/data/Roberta_100_Short_Wiki.jsonl", 'a')
  search_results = []
  api_results =  google_search_api(search_term = all_entities[idx], api_key = API_KEY,
                                  cse_id = SEID,
                                  num_results = 5,num_iters = 1)
  search_results.append(api_results)
  extracted_urls = get_urls_api(search_results)
  # Push wikipedia to the top (if it exists)
  extracted_urls = list(extracted_urls)
  for i, url in enumerate(extracted_urls):
    if "wikipedia" in url:
      extracted_urls.insert(0, extracted_urls.pop(i))      
  article_num = 0
  all_articles = {}
  all_urls = {}
  for link in tqdm(extracted_urls):
    news = newspaper(link)
    if (len(news.article.split(' ')) <= 100):
      continue
    else:
      art = news.get_dict['article']
      # Cleaning the article
      art_split = art.split('.')
      art_len = len(art_split) - 1
      for i, sent in enumerate(reversed(art_split)):
          if('disambiguation' in sent.lower() or 'redirects' in sent.lower()):
              art_split.pop(art_len - i)      
      art = '. '.join(art_split)
      # Removing wikipedia based artifacts
      art = re.sub(' ?\[[0-9]*\]', "", art)
      art = re.sub('\..{1,40}\[ edit \]', ".", art)
      art = re.sub('\[ edit \]', "", art)
      all_articles[f'article_{article_num}'] = art
      all_urls[f'URL_{article_num}'] = link
      article_num+=1  
  out_dict = {"Index" : idx, "Fact" : fact, "Entity" : all_entities[idx], "URLS" : all_urls, 'Articles': all_articles}
  out_file.write(out_dict)
  out_file.close()











