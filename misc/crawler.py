import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime as dt
from randomheaders import LoadHeader as random_header


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


random.seed(123)
COLUMNS = ['time', 'title']
INFO = pd.DataFrame(columns=COLUMNS)
IP_LIST = []


def get_proxy():
    return requests.get('http://localhost:5555/get/').text

def delete_proxy(proxy):
    requests.get(f'http://localhost:5555/delete/?proxy={proxy}')

def get_soup(url, parser):
    while True:
        try:
            time.sleep(random.uniform(0, 1))
            proxy = get_proxy()
            html = requests.get(
                url, timeout=(3, 7), headers=random_header(),
                proxies={'http': proxy, 'https': proxy}).text
            return BeautifulSoup(html, parser)
        except Exception:
            delete_proxy(proxy)

def parse_time(timestr):
    raw_time = timestr[10:].split('/')[0].replace('on', '')
    time = dt.strptime(raw_time, '%H:%M  %B %d, %Y  ')
    return time

def get_content(url):
    soup = get_soup(url, 'lxml')
    paragraphs = soup.select('div[class="entry-content"]')[0].select('p')
    content = '\n'.join([para.text for para in paragraphs[:-1]])
    return content

def get_posts(soup):
    if '404 - PAGE NOT FOUND' in soup.text: return None
    posts = soup.select('a[class="stream-article"]')
    titles, times, abstracts, contents = [], [], [], []
    for p in posts:
        meta = p.select('div[class="meta"]')[0]
        titles.append(meta.select('h3')[0].text)
        times.append(meta.select('time')[0]['datetime'])
        abstracts.append(meta.select('p')[0].text)
        contents.append(get_content(p['href']))
    posts = pd.DataFrame({'title': titles, 'abstract': abstracts, 'content': contents}, index=times)
    posts.index = pd.to_datetime(posts.index)
    return posts


all_posts = pd.DataFrame(columns=['title', 'abstract', 'content'])

i = 4
while True:
    try:
        i += 1
        print(f'Crawling page {i}: ', end='')
        url = f'https://www.coindesk.com/category/markets-news/page/{i}'
        soup = get_soup(url, 'lxml')
        posts = get_posts(soup)
        if posts is None: raise EOFError
        all_posts = pd.concat([all_posts, posts])
        print(f'finished ({posts.shape[0]} posts)')
    except IndexError:
        print('failed, retrying', end='\r')
        i -= 1
    except (EOFError, KeyboardInterrupt):
        print('terminated')
        print(f'Total number of posts crawled: {all_posts.shape[0]}')
        all_posts.to_csv('../data/posts.csv')
        exit()
