
import json
from concurrent.futures import as_completed

import numpy as np
import requests
from bs4 import BeautifulSoup
from requests_futures.sessions import FuturesSession
from tqdm import tqdm

from constants import TMDB_API_KEY
from plotting_utils import transform_ratings


def get_ratings_from_futures(soup):
    # grab the main film grid
    table = soup.find("ul", class_="poster-list")

    if table is None:
        return None

    films = table.find_all("li")

    film_ratings = list()

    for film in films:
        stars = transform_ratings(
            film.find("p", class_="poster-viewingdata").get_text().strip()
        )
        film_link = film.find_all("div")[0].get("data-film-slug")

        if stars == -1:
            continue
        film_ratings.append((film.find("div").find("img")["alt"], stars, film_link))

    if len(film_ratings) == 0:
        return None

    return film_ratings


def get_diary_links(user_name):
    rlink = f"https://letterboxd.com/{user_name}/films/diary/"
    ratings_page = requests.get(rlink)

    if ratings_page.status_code != 200:
        print("PAGE NOT FOUND")
        return 0

    soup = BeautifulSoup(ratings_page.content, "lxml")
    try:
        num_pages = int(
            soup.find("div", class_="pagination").find_all("a")[-1].contents[0]
        )
        url_list = [rlink] + [rlink + f"page/{idx}/" for idx in range(2, num_pages + 1)]
    except:
        url_list = [rlink]
    return url_list


def get_links(user_name):
    rlink = f"https://letterboxd.com/{user_name}/films/"
    ratings_page = requests.get(rlink)

    if ratings_page.status_code != 200:
        print("PAGE NOT FOUND")
        return 0

    soup = BeautifulSoup(ratings_page.content, "lxml")
    try:
        num_pages = int(
            soup.find("div", class_="pagination").find_all("a")[-1].contents[0]
        )
        url_list = [rlink] + [rlink + f"page/{idx}/" for idx in range(2, num_pages + 1)]
    except:
        url_list = [rlink]
    return url_list


def get_poster_link(movie_links):
    with FuturesSession() as session:
        futures = [session.get(ele) for ele in tqdm(movie_links)]
        poster_links = dict()
        for future in as_completed(futures):
            temp = future.result()
            if temp.status_code != 200:
                print("PAGE NOT FOUND")
                poster_links[temp.url] = None
                continue
            soup = BeautifulSoup(temp.content, "lxml")
            # panel = soup.find('div', class_="film-poster").find('img')
            panel = soup.find(string="TMDb").find_parent("a").get("href")

            tmdb_id = panel.split("/")[-2]
            tmbd_response = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
            )
            tmbd_response = json.loads(tmbd_response.content)
            if "success" in tmbd_response.keys():
                poster_links[temp.url] = None
            else:
                poster_links[temp.url] = (
                    "https://image.tmdb.org/t/p/original" + tmbd_response["poster_path"]
                )
    return poster_links


def get_film_data(film_soup, film_url, str_idx, section_placeholder, meta_data_dict):
    str_0 = (
        ""
        if meta_data_dict["len_urls"] < 500
        else ". This might take a while. Till then look at some stats that we found about your profile..."
    )
    dyk_str = "## *Did you know?*\n"
    idx_msgs = {
        0: f"## Wow! you have rated {meta_data_dict['len_ratings']} films" + str_0,
        250: dyk_str
        + f"## You rated {meta_data_dict['nfilms_this_month']} films this month and\
          {meta_data_dict['nfilms_this_year']} this year",
        900: dyk_str
        + f"## Last year, you rated the most films in \
        {meta_data_dict['nfilms_last_year_most_month']} ({meta_data_dict['nfilms_last_year_most_month_count']}) \
                    and rated films the highest in {meta_data_dict['nfilms_last_year_most_rated_month']}\
                          ({meta_data_dict['nfilms_last_year_most_rated_month_val']}).",
        1500: dyk_str
        + f"## You usually rate films on Letterboxd in the {meta_data_dict['nfilms_time_of_day']}.",
        2000: dyk_str
        + f"## You rated the most films ({meta_data_dict['nfilms_most_year_count']} films)\
          in {meta_data_dict['nfilms_most_year']}.",
        2500: dyk_str
        + f"## {meta_data_dict['first_film_name']} is one of the first films you rated on Letterboxd!\
          You gave it {meta_data_dict['first_film_rating']}/5",
        3500: f"## Do you feel like reassessing your recent {meta_data_dict['latest_film_rating']}\
          rating for {meta_data_dict['latest_film_name']}?",
    }

    if str_idx in idx_msgs.keys():
        section_placeholder.empty()
        section_placeholder.markdown(f"{idx_msgs[str_idx]}")
    elif str_idx - 3500 in idx_msgs.keys():
        section_placeholder.empty()
        section_placeholder.markdown(f"{idx_msgs[str_idx - 3500]}")

    # average rating
    try:
        avg_rating = film_soup.find("meta", attrs={"name": "twitter:data2"}).attrs[
            "content"
        ]
        avg_rating = float(avg_rating.split(" ")[0])
    except:
        avg_rating = np.nan
    # year
    try:
        yr_section = (
            film_soup.find("section", attrs={"id": "featured-film-header"})
            .find("small")
            .find("a")
        )
        year = int(yr_section.contents[0])
    except:
        year = np.nan

    # director
    try:
        director = str(
            film_soup.find("meta", attrs={"name": "twitter:data1"}).attrs["content"]
        )
    except:
        director = ""

        # print("COUNTRIES")
    no_details = False
    countries = np.nan
    langs = np.nan
    genres = []
    try:
        span = film_soup.find("div", attrs={"id": "tab-details"}).select("span")
    except:
        no_details = True

    if not no_details:
        countries = []
        langs = []
        for s in span:
            if s.contents[0] == "Countries" or s.contents[0] == "Country":
                d1 = s.find_next("div")
                countries = [str(c.contents[0]) for c in d1.find_all("a")]
                # countries.append(str(c.contents[0]))
            if s.contents[0] == "Languages" or s.contents[0] == "Language":
                d1 = s.find_next("div")
                langs = [str(c.contents[0]) for c in d1.find_all("a")]
                # langs.append(str(c.contents[0]))
        if len(countries) == 0:
            countries = np.nan
        if len(langs) == 0:
            langs = np.nan

    # GENRES THEMES
    no_genre = False
    genres = np.nan
    th_list = np.nan
    try:
        span = film_soup.find("div", attrs={"id": "tab-genres"}).select("span")
    except:
        no_genre = True

    if not no_genre:
        genres = []
        th_list = []
        for s in span:
            if s.contents[0] == "Genres" or s.contents[0] == "Genre":
                d1 = s.find_next("div")
                genres = [
                    (str(c.contents[0]), str(c["href"]))
                    for c in d1.find_all("a", href=True)
                ]
            if s.contents[0] == "Themes" or s.contents[0] == "Theme":
                d1 = s.find_next("div")
                th_list = [
                    (str(c.contents[0]), str(c["href"]))
                    for c in d1.find_all("a", href=True)
                ]
        if len(genres) == 0:
            genres = np.nan
        if len(th_list) == 0:
            th_list = np.nan

    return [film_url, year, director, avg_rating, countries, langs, genres, th_list]
