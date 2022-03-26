import asyncio
import requests_async as requests_a
import time
from tqdm import tqdm
import platform
import numpy as np

from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
import requests
from bs4 import BeautifulSoup
from requests_async.exceptions import HTTPError, RequestException, Timeout
import cchardet
import lxml
from datetime import datetime

_domain = "https://letterboxd.com"

def transform_ratings(some_str):
    """
    transforms raw star rating into float value
    :param: some_str: actual star rating
    :rtype: returns the float representation of the given star(s)
    """
    stars = {
        "★": 1,
        "★★": 2,
        "★★★": 3,
        "★★★★": 4,
        "★★★★★": 5,
        "½": 0.5,
        "★½": 1.5,
        "★★½": 2.5,
        "★★★½": 3.5,
        "★★★★½": 4.5
    }
    try:
        return stars[some_str]
    except:
        return -1


def get_ratings_from_futures(ratings_page):
  # check to see page was downloaded correctly
  if ratings_page.status_code != 200:
      print("PAGE NOT FOUND")
      return None

  # t1=time.time()
  soup = BeautifulSoup(ratings_page.content, 'lxml')
  # print(f"Parsing time {time.time()-t1}")

  # grab the main film grid
  # t1=time.time()
  table = soup.find('ul', class_='poster-list')
  # print(f"table time {time.time()-t1}")
  if table is None:
      return None

  # t1=time.time()
  films = table.find_all('li')
  # print(f"film find_all time {time.time()-t1}")
  
  film_ratings = list()

  # iterate through friends
  # t1=time.time()
  for film in films:
      stars = transform_ratings(film.find('p', class_='poster-viewingdata').get_text().strip())
      film_link = _domain + film.find('div').get('data-target-link')
      rated_date = film.find('p', class_='poster-viewingdata').find('time')['datetime']
      rated_date = datetime.strptime(rated_date[:-1], "%Y-%m-%dT%H:%M:%S")
      if stars == -1:
          continue
      film_ratings.append((film.find('div').find('img')['alt'], stars, rated_date, film_link))
  
  # print(f"film iterations time {time.time()-t1}")
  
  return film_ratings

def get_links():
    user_name="arjunrajput"
    rlink = f"https://letterboxd.com/{user_name}/films/ratings/"
    ratings_page = requests.get(rlink)

    if ratings_page.status_code != 200:
      print("PAGE NOT FOUND")
      return 0

    soup = BeautifulSoup(ratings_page.content, 'lxml')
    try:
      num_pages=int(soup.find('div', class_="pagination").find_all('a')[-1].contents[0])
      url_list = [rlink] + [rlink+f"page/{idx}/" for idx in range(2,num_pages+1)]
    except:
        url_list = [rlink]
    return url_list

    # with FuturesSession() as session:
    #     t1=time.time()
    #     futures = [session.get(ele) for ele in tqdm(url_list)]
    #     ratings = []
    #     for future in as_completed(futures):
    #         temp = get_ratings_from_futures(future.result())
    #         if temp is None:
    #             return 0
    #         ratings = ratings + temp

    # print(len(ratings), time.time()-t1)
    # return [ele[-1] for ele in ratings]    


def get_film_data(filmget):
  # film_page="https://letterboxd.com/film/phantom-thread/"
  # filmget = requests.get(film_page)
  # session = requests.Session()
  # filmget = session.get(film_page)

  film_soup = BeautifulSoup(filmget.content, 'lxml')
  
  #average rating
  try:
    avg_rating = film_soup.find('meta', attrs={'name':'twitter:data2'}).attrs['content']
    avg_rating = float(avg_rating.split(' ')[0])
  except:
    avg_rating = np.nan
  #year
  try:
    yr_section = film_soup.find('section', attrs={'id':'featured-film-header'}).find('small').find('a')
    year = int(yr_section.contents[0])
  except:
    year = np.nan

  #director
  try:
    director = str(film_soup.find('meta', attrs={'name':'twitter:data1'}).attrs['content'])
  except:
    director = ''        

  # print("COUNTRIES")
  no_details = False
  countries=np.nan
  langs=np.nan
  genres=[]
  try:
    span = film_soup.find('div', attrs={'id':'tab-details'}).select("span")
  except:
    no_details = True

  if not no_details:
    countries=[]
    langs=[]
    for s in span:
      if s.contents[0]=="Countries" or s.contents[0]=="Country":
        d1 = s.find_next('div')
        countries = [str(c.contents[0]) for c in d1.find_all('a')]
        # countries.append(str(c.contents[0]))
      if s.contents[0]=="Languages" or s.contents[0]=="Language":
        d1 = s.find_next('div')
        langs = [str(c.contents[0]) for c in d1.find_all('a')]
        # langs.append(str(c.contents[0]))
    if len(countries)==0:
      countries=np.nan
    if len(langs)==0:
      langs=np.nan
    
  try:
    cast = [ line.contents[0] for line in film_soup.find('div', attrs={'id':'tab-cast'}).find_all('a')]
    # remove all the 'Show All...' tags if they are present
    cast = [str(i) for i in cast if i != 'Show All…']
    cast = cast[:10]
  except:
    cast = np.nan

  #GENRES THEMES
  no_genre = False
  genres=np.nan
  th_list=np.nan
  try:
    span = film_soup.find('div', attrs={'id':'tab-genres'}).select("span")
  except:
    no_genre = True

  if not no_genre:
    genres=[]
    th_list=[]
    for s in span:
      if s.contents[0]=="Genres" or s.contents[0]=="Genre":
        d1 = s.find_next('div')
        genres = [(str(c.contents[0]), str(c['href'])) for c in d1.find_all('a', href=True)]
        # genres.append((str(c.contents[0]), str(c['href'])))
      if s.contents[0]=="Themes" or s.contents[0]=="Theme":
        d1 = s.find_next('div')
        th_list = [(str(c.contents[0]), str(c['href'])) for c in d1.find_all('a', href=True)]
        # th_list.append((str(c.contents[0]), str(c['href'])))
    if len(genres)==0:
      genres=np.nan
    if len(th_list)==0:
      th_list=np.nan
  
  return [filmget.url, year, director, avg_rating, countries, langs, genres, th_list, cast]    


MAX_REQUESTS = 200
# URLS = get_links()

class BaseException(Exception):
    pass


class HTTPRequestFailed(BaseException):
    pass


async def fetch(url, timeout=None):
    async with requests_a.Session() as session:
        try:
            resp = await session.get(url, timeout=timeout)
            resp.raise_for_status()
        except HTTPError:
            raise HTTPRequestFailed(f'Skipped: {resp.url} ({resp.status_code})')
        except Timeout:
            raise HTTPRequestFailed(f'Timeout: {url}')
        except RequestException as e:
            raise HTTPRequestFailed(e)
    return resp


async def parse_html(html):
    film_soup = BeautifulSoup(html, 'lxml')
    return film_soup

async def run(sem, url):
    async with sem:
        start_t = time.time()
        resp = await fetch(url)
        # film_soup = await parse_html(resp.content)
        end_t = time.time()
        elapsed_t = end_t - start_t
        r_time = resp.elapsed.total_seconds()
        
        return resp

async def get_links_async():
    sem = asyncio.Semaphore(MAX_REQUESTS)
    user_rating_pages = get_links()

    tasks = [asyncio.create_task(run(sem, url)) for url in tqdm(user_rating_pages)]
    film_link_pages=[]
    # idx=0
    t1=time.time()
    # for f in asyncio.as_completed(tasks):
    #     rating_page_response = await f
        
    #     film_link_pages=film_link_pages+[ele[-1] for ele in get_ratings_from_futures(rating_page_response)]
    
    film_rating_details=[ele for f in asyncio.as_completed(tasks) for ele in get_ratings_from_futures(await f)]

    film_link_pages=[ele[-1] for ele in film_rating_details]

    print(f"{len(film_link_pages)} in {time.time()-t1}, {film_rating_details[0]}")

    return film_link_pages, film_rating_details

async def main(URLS):
    sem = asyncio.Semaphore(MAX_REQUESTS)
    tasks = [asyncio.create_task(run(sem, url)) for url in tqdm(URLS)]
    result_pages=[]
    
    t1=time.time()
    
    result_pages=[get_film_data(await f) for f in asyncio.as_completed(tasks)]
    
    print(f"{len(result_pages)} in {time.time()-t1}, {result_pages[0]}")

    return result_pages


if __name__=="__main__":
    if 'Windows' in platform.system():
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    film_link_pages, film_rating_details = asyncio.run(get_links_async())
    ffd = asyncio.run(main(film_link_pages))
    # await main()