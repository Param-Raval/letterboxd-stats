import ast
import asyncio
import json
import platform
import time
import traceback
from datetime import datetime

import aiohttp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bokeh.models import ColumnDataSource, HoverTool, OpenURL, TapTool
from bokeh.models.tools import TapTool
from bokeh.plotting import figure
from bs4 import BeautifulSoup
from requests_async.exceptions import RequestException, Timeout
from tqdm import tqdm

from constants import MAX_REQUESTS, _domain
from page_parsing import (
    get_diary_links,
    get_film_data,
    get_links,
    get_poster_link,
    get_ratings_from_futures,
)
from plotting_utils import (
    get_image_from_url,
    make_bar_chart,
    make_horizontal_bar_chart,
    make_world_map,
)
from utils import get_meta_dict, open_link

st.set_page_config(layout="wide")


class BaseException(Exception):
    pass


class HTTPRequestFailed(BaseException):
    pass


async def fetch(url, timeout=None):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as resp:
                resp.raise_for_status()
                body = await resp.text()
                soup = BeautifulSoup(body, "lxml")
                return soup
        except Timeout:
            raise HTTPRequestFailed(f"Timeout: {url}")
        except RequestException as e:
            raise HTTPRequestFailed(e)
        except aiohttp.ClientError as e:
            raise HTTPRequestFailed(f"Failed to fetch {url}: {e}")


async def parse_html(html):
    film_soup = BeautifulSoup(html, "lxml")
    return film_soup


async def run(sem, url):
    async with sem:
        return await fetch(url)


async def get_links_async(user_name):
    sem = asyncio.Semaphore(MAX_REQUESTS)
    user_rating_pages = get_links(user_name=user_name)
    user_diary_pages = get_diary_links(user_name=user_name)

    tasks = [asyncio.create_task(run(sem, url)) for url in tqdm(user_rating_pages)]

    diary_tasks = [asyncio.create_task(run(sem, url)) for url in tqdm(user_diary_pages)]

    film_link_pages = []
    t1 = time.time()
    if len(tasks) == 0:
        return None, None, None

    try:
        film_rating_details = [
            ele
            for f in asyncio.as_completed(tasks)
            for ele in get_ratings_from_futures(await f)
            if ele is not None
        ]
        film_diary_dates = [
            ele
            for f in asyncio.as_completed(diary_tasks)
            for ele in get_dates_from_diary(await f)
            if ele is not None
        ]
    except Exception as e:
        traceback.print_exc()
        return None, None, None

    if len(film_rating_details) == 0:
        print("No ratings found!")
        return None, None, None
    film_link_pages = [ele[-1] for ele in film_rating_details]

    print(f"{len(film_link_pages)} in {time.time() - t1}")

    return film_link_pages, film_rating_details, film_diary_dates


def get_dates_from_diary(soup):

    films = soup.find_all("tr", class_="diary-entry-row")

    rated_dates = []

    for film in films:
        date_link = film.find("td", class_="td-day").find("a", href=True)["href"]
        diary_year = str(date_link).split("/")[-4]
        diary_month = str(date_link).split("/")[-3]
        diary_day = str(date_link).split("/")[-2]
        rated_date = datetime.strptime(
            f"{diary_year}-{diary_month}-{diary_day}", "%Y-%m-%d"
        )

        film_link = film.find("td", class_="td-film-details").find_all("div")[0][
            "data-film-slug"
        ]

        rated_dates.append((rated_date, film_link))

    return rated_dates


async def get_film_main(urls, meta_data_dict):
    sem = asyncio.Semaphore(MAX_REQUESTS)
    tasks = [asyncio.create_task(run(sem, url)) for url in tqdm(urls)]
    result_pages = []

    t1 = time.time()

    section_placeholder = st.empty()

    result_pages = [
        get_film_data(
            await f,
            urls[idx],
            idx + meta_data_dict["current_batch_idx"],
            section_placeholder,
            meta_data_dict,
        )
        for f, idx in zip(asyncio.as_completed(tasks), range(0, len(urls)))
    ]

    section_placeholder.empty()

    print(f"{len(result_pages)} in {time.time() - t1}")

    return result_pages


def get_film_df(user_name):
    film_cache = pd.read_excel("./film_cache.xlsx")

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

    print(f"Found {len(url_list)} page/s.")
    print(f"Running on {platform.system()}.")
    if "Windows" in platform.system():
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    films_url_list, ratings, diary_dates = asyncio.run(get_links_async(user_name))

    films_url_list = [f"{_domain}/film/" + link for link in films_url_list]

    if ratings is None:
        return None

    st.markdown(
        f"##### 2. Fetched {len(ratings)} ratings from {user_name}! Fetching movie data..."
    )

    films_url_list_new = [
        film_url
        for film_url in films_url_list
        if film_url not in film_cache["lbxd_link"].values.tolist()
    ]

    max_new_fetch_limit = 2500
    if len(films_url_list_new) > max_new_fetch_limit:
        st.markdown(
            f"### Wow! you have rated {len(films_url_list)} films.\
                For now, this is too much for us to process. Processing\
                 {len(films_url_list) - len(films_url_list_new) + max_new_fetch_limit} films..."
        )
        films_url_list_new = films_url_list_new[:max_new_fetch_limit]

    ratings_df = pd.DataFrame(ratings, columns=["film_name", "rating", "lbxd_link"])
    dates_df = pd.DataFrame(diary_dates, columns=["rated_date", "lbxd_link"])

    dates_df["lbxd_link"] = dates_df["lbxd_link"].astype("str").replace("NaN", pd.NA)

    dates_df.dropna(subset=["lbxd_link"], inplace=True)

    ratings_df = ratings_df.merge(dates_df, how="inner", on="lbxd_link")

    ratings_df["lbxd_link"] = ratings_df["lbxd_link"].apply(
        lambda row: f"{_domain}/film/" + row
    )

    # compute prelimnary rating based stats
    meta_data_dict = get_meta_dict(ratings_df)
    meta_data_dict["len_urls"] = len(films_url_list_new)
    meta_data_dict["len_ratings"] = len(films_url_list)

    ffd = list()
    fetch_batch_size = 1000
    for batch_idx in range(0, len(films_url_list_new), fetch_batch_size):
        films_url_batch = films_url_list_new[batch_idx : batch_idx + fetch_batch_size]
        meta_data_dict["current_batch_idx"] = batch_idx
        ffd = ffd + asyncio.run(get_film_main(films_url_batch, meta_data_dict))

    # st.markdown(f"##### 3. Fetched {len(ratings)} movies!")
    print(
        f"Fetched {len(ffd)} movies! {len(films_url_list) - len(films_url_list_new)} found in cache."
    )

    # all ratings of the user
    ratings_df = ratings_df.drop("rated_date", axis=1)

    candidates = ["rated_date_year", "rated_date_month", "rated_date_time_day"]
    drop_columns = [x for x in candidates if x in ratings_df.columns]
    if len(drop_columns) > 0:
        ratings_df = ratings_df.drop(drop_columns, axis=1)

    # only newly fetched film data
    column_list = [
        "lbxd_link",
        "year",
        "director",
        "avg_rating",
        "countries",
        "langs",
        "genres",
        "themes",
    ]
    film_df = pd.DataFrame(ffd, columns=column_list)

    # ratings and film data of newly fetched movies
    # df = ratings_df.join(film_df.set_index('lbxd_link'), on='lbxd_link')

    # ratings only of newly fetched movies
    ratings_df_new = ratings_df[ratings_df["lbxd_link"].isin(films_url_list_new)]
    df_new = ratings_df_new.join(film_df.set_index("lbxd_link"), on="lbxd_link")
    new_film_cache = pd.concat(
        [df_new.drop("rating", axis=1), film_cache], ignore_index=True
    )
    new_film_cache.to_excel("./film_cache.xlsx", index=False)
    new_film_cache = None
    film_df = None

    # film cache of user's films
    int_film_cache = film_cache[
        film_cache["lbxd_link"].isin(
            [
                film_url
                for film_url in films_url_list
                if film_url in film_cache["lbxd_link"].values.tolist()
            ]
        )
    ]

    film_cache = None

    # add user's ratings to existing film cache
    ratings_df_film_cache = ratings_df.join(
        int_film_cache.drop("film_name", axis=1).set_index("lbxd_link"),
        on="lbxd_link",
        how="inner",
    )
    int_film_cache = None

    ratings_df_film_cache["genres"] = ratings_df_film_cache["genres"].apply(
        lambda x: ast.literal_eval(x) if x != "nan" and x is not np.nan else np.nan
    )
    ratings_df_film_cache["themes"] = ratings_df_film_cache["themes"].apply(
        lambda x: ast.literal_eval(x) if x != "nan" and x is not np.nan else np.nan
    )
    ratings_df_film_cache["langs"] = ratings_df_film_cache["langs"].apply(
        lambda x: ast.literal_eval(x) if x != "nan" and x is not np.nan else np.nan
    )
    ratings_df_film_cache["countries"] = ratings_df_film_cache["countries"].apply(
        lambda x: ast.literal_eval(x) if x != "nan" and x is not np.nan else np.nan
    )

    # add new ratings+film data to existing ratings+film cache
    return pd.concat([df_new, ratings_df_film_cache], ignore_index=True)


def main():
    # Main page
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 2, 0.2, 1, 0.1)
    )

    row0_1.title("Analyzing your Letterboxd Profile")

    with row0_2:
        st.write("")

    row0_2.subheader("A web app by Param Raval")

    row01_1, row01_2, row01_3, row01_spacer3 = st.columns((5, 0.5, 1, 0.5))

    row01_1.markdown("""---""")
    row01_2.button(
        "LinkedIn",
        on_click=open_link,
        args=(("https://www.linkedin.com/in/param-raval/",)),
    )
    row01_3.button(
        "Paradise Cinema",
        on_click=open_link,
        args=(("https://paradisecinemaa.wordpress.com/",)),
    )

    row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

    with row1_1:
        st.markdown(
            '#### <h5><b> To begin, please enter your <a href=https://www.letterboxd.com/ \
            style="color: #b9babd; text-decoration: underline;">Letterboxd</a> **profile name\
              (or just use mine!).** 👇 </b></h5>',
            unsafe_allow_html=True,
        )

    row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns(
        [0.1, 2, 0.2, 1, 0.1]
    )
    with row2_1:
        default_username = "param_raval"
        user_input = st.text_input(
            "Input your own Letterboxd profile name (e.g. param_raval)", "param_raval"
        )
        need_help = st.expander("Need help? 👉")
        with need_help:
            st.markdown(
                "Having trouble finding your Letterboxd profile? Head to the\
                      [Letterboxd website](https://www.letterboxd.com/) and click on your profile at the top."
            )

        if not user_input:
            user_input = f"{default_username}"

        bt1 = st.button("Go!")

    with row2_2:
        st.subheader(" ")
        st.markdown(
            f'<p style="background-color:#101010;color:#101010;font-size:24px;">--</p>',
            unsafe_allow_html=True,
        )

    if not bt1:
        st.markdown(f"")
    else:
        if user_input == default_username:
            df = pd.read_excel("./letterboxd_film_data1.xlsx")
        else:
            with st.spinner(text="Good things come to those who wait..."):
                st.markdown(f"##### 1. Fetching {user_input}'s profile")
                df = get_film_df(user_input)
                if isinstance(df, int) and df == 0:
                    st.markdown(f"### Username {user_input} not found!")
                    return 0
                if df is None:
                    st.markdown(f"### {user_input} has not rated any films!")
                    return 0

        st.markdown(f"## Analyzing {user_input}'s profile")
        st.markdown("""---""")

        st.subheader("You explored films from a spectrum of years")
        st.markdown(f"#### Number of films you rated across release years")

        df = df[~(df["year"] == np.nan) & (df["year"].notna())]
        df = df[~(df["film_name"] == np.nan) & (df["film_name"].notna())]

        df_count = df[["film_name", "year"]].groupby(by=["year"]).count()
        df_count["year"] = df_count.index

        df_count["decade"] = (
            df_count["year"].apply(lambda row: row - row % 10).astype(int)
        )
        df_count["year_rating"] = (
            df[["rating", "year"]].groupby(by=["year"]).mean()["rating"]
        )

        df["decade"] = df["year"].apply(lambda row: row - row % 10).astype(int)

        df_count2 = df[["film_name", "decade"]].groupby(by=["decade"]).count()
        df_count2["decade"] = df_count2.index
        df_count2["decade_rating"] = (
            df_count[["year_rating", "decade"]]
            .groupby(by=["decade"])
            .mean()["year_rating"]
        )

        if len(df_count) == 0:
            st.markdown("#### Not enough data to show")
        else:
            fig = make_bar_chart(
                data=df_count, x="year", y="film_name", color="year", ccs="sunset"
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        st.markdown(f"#### Average ratings you gave to each year")

        if len(df_count) == 0:
            st.markdown("#### Not enough data to show")
        else:
            fig = make_bar_chart(
                data=df_count, x="year", y="year_rating", color="year", ccs="viridis"
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        st.subheader("Decades in film!")
        st.markdown(f"#### Decades of films watched by you!")

        if len(df_count2) == 0:
            st.markdown("#### Not enough data to show")
        else:
            fig = make_bar_chart(
                data=df_count2, x="decade", y="film_name", color="decade", ccs="emrld"
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        st.markdown(f"#### Average ratings you gave to each decade of film...")

        if len(df_count2) == 0:
            st.markdown("#### Not enough data to show")
        else:
            fig = make_bar_chart(
                data=df_count2,
                x="decade",
                y="decade_rating",
                color="decade",
                ccs="agsunset",
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        st.markdown(f"### Your favourite decades")
        st.markdown(f"##### ...and a handful of your favourites from them")

        hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
                    """

        lbxd_list = []
        v = df_count["decade"].value_counts()
        df_decade_cnt_filtered = df_count[["year_rating", "decade"]][
            (df_count["decade"].isin(v[v > 2].index.values.tolist()))
        ]

        df_decade_cnt_filtered["decade_rating"] = (
            df_decade_cnt_filtered[["year_rating", "decade"]]
            .groupby(by=["decade"])
            .mean()["year_rating"]
        )

        for name, group in df_count[
            (
                df_count["decade"].isin(
                    df_decade_cnt_filtered.nlargest(3, "decade_rating")[
                        "decade"
                    ].values.tolist()
                )
            )
        ].groupby(by=["decade"]):
            df_group = (
                df[["rating", "lbxd_link"]][
                    df["year"].isin(group["year"].values.tolist())
                ]
                .nlargest(5, "rating")
                .reset_index(drop=True)
                .drop("rating", axis=1)
            )
            lbxd_list = lbxd_list + df_group["lbxd_link"].values.tolist()

        if len(lbxd_list) == 0:
            st.markdown("#### Not enough data to show.")

        poster_links = get_poster_link(lbxd_list)

        for name, group in df_count[
            (
                df_count["decade"].isin(
                    df_decade_cnt_filtered.nlargest(3, "decade_rating")[
                        "decade"
                    ].values.tolist()
                )
            )
        ].groupby(by=["decade"]):
            need_help = st.expander(f"{name}s")
            with need_help:
                df_group = (
                    df[["film_name", "year", "rating", "lbxd_link"]][
                        df["year"].isin(group["year"].values.tolist())
                    ]
                    .nlargest(5, "rating")
                    .reset_index(drop=True)
                )
                num_movies_in_group = len(df_group)

                film_poster_rows = st.columns((10, 10, 10, 10, 10))
                blank_spots = len(film_poster_rows) - len(df_group)

                df_group = df_group.reindex(
                    list(range(0, len(film_poster_rows)))
                ).reset_index(drop=True)

                cnt = 0
                empty_poster_url = (
                    "https://s.ltrbxd.com/static/img/empty-poster-110.e0cbb286.png"
                )
                for poster_row, (idx, row) in zip(
                    film_poster_rows, df_group.iterrows()
                ):
                    cnt = cnt + 1
                    with poster_row:
                        if cnt > num_movies_in_group and blank_spots > 0:
                            st.markdown(" ")
                        else:
                            try:
                                st.bokeh_chart(
                                    get_image_from_url(
                                        poster_links[row["lbxd_link"]], row
                                    )
                                )
                            except:
                                st.bokeh_chart(
                                    get_image_from_url(empty_poster_url, row)
                                )

        st.markdown("Posters from [TMDb](https://www.themoviedb.org/)")
        df_count = None
        st.markdown(f"\n")
        st.markdown("""---""")
        st.markdown("## Find yourself in the films you watch")

        (
            row3_space1,
            row3_1,
            row3_space2,
            row3_2,
            row3_space3,
            row3_3,
            row4_space4,
        ) = st.columns((0.1, 15, 0.1, 15, 0.1, 15, 1.0))
        with row3_1:
            st.subheader("Genres")
            st.markdown(f"#### that you __watched__ the most...")

            df_genre = df[["film_name", "genres"]]

            df_genre = (
                df_genre[~(df_genre["genres"] == np.nan) & (df_genre["genres"].notna())]
                .reset_index()
                .drop("index", axis=1)
            )

            if isinstance(df_genre["genres"].values.tolist()[0], str):
                df_genre["genres"] = df_genre["genres"].apply(
                    lambda x: ast.literal_eval(x)
                )

            df_genre["only_genre"] = df_genre["genres"]

            df_genre["only_genre"] = [
                [e[0] for e in ele] if ele is not np.nan else ele
                for ele in df_genre["only_genre"].values.tolist()
            ]
            df_genre = (
                df_genre[df_genre["genres"].notna()].reset_index().drop("index", axis=1)
            )
            df_genre = df_genre["only_genre"].explode().reset_index()
            df_genre["index"] = df_genre.index

            df_genre_cnt = (
                df_genre.groupby(by=["only_genre"])
                .count()
                .sort_values(by=["index"], ascending=False)
            )
            df_genre_cnt["only_genre"] = df_genre_cnt.index
            df_genre_cnt = df_genre_cnt.nlargest(10, "index")

            if len(df_genre_cnt) == 0:
                st.markdown("")

            else:
                fig = make_horizontal_bar_chart(
                    data=df_genre_cnt, y="only_genre", x="index", ccs="lightseagreen"
                )

                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

            st.markdown(f"#### that you *loved* the most...")

            df_genre_rating = df[["rating", "genres"]]
            df_genre_rating = (
                df_genre_rating[
                    ~(df_genre_rating["genres"] == np.nan)
                    & (df_genre_rating["genres"].notna())
                ]
                .reset_index()
                .drop("index", axis=1)
            )

            if isinstance(df_genre_rating["genres"].values.tolist()[0], str):
                df_genre_rating["genres"] = df_genre_rating["genres"].apply(
                    lambda x: ast.literal_eval(x)
                )

            df_genre_rating["only_genre"] = df_genre_rating["genres"]

            df_genre_rating["only_genre"] = [
                [e[0] for e in ele] if ele is not np.nan else ele
                for ele in df_genre_rating["only_genre"].values.tolist()
            ]
            df_genre_rating = (
                df_genre_rating[df_genre_rating["genres"].notna()]
                .reset_index()
                .drop("index", axis=1)
            )
            df_genre_rating.index = df_genre_rating["rating"]
            df_genre_rating = df_genre_rating["only_genre"].explode()

            df_genre_rating2 = pd.DataFrame(columns=["rating", "only_genre"])
            df_genre_rating2["rating"] = df_genre_rating.index.values.tolist()
            df_genre_rating2["only_genre"] = df_genre_rating.values.tolist()

            v = df_genre_rating2["only_genre"].value_counts()
            df_genre_rating2 = df_genre_rating2[
                df_genre_rating2["only_genre"].isin(v[v > 5].index.values.tolist())
            ]

            df_genre_rating_cnt = (
                df_genre_rating2.groupby(by=["only_genre"])
                .mean()
                .sort_values(by=["rating"], ascending=False)
            )
            df_genre_rating_cnt = df_genre_rating_cnt.nlargest(10, "rating")
            df_genre_rating_cnt["only_genre"] = df_genre_rating_cnt.index
            df_genre_rating_cnt.reset_index(drop=True, inplace=True)

            if len(df_genre_rating_cnt) == 0:
                st.markdown("#### Not enough data to show.")

            else:
                fig = make_horizontal_bar_chart(
                    data=df_genre_rating_cnt,
                    y="only_genre",
                    x="rating",
                    ccs="lightseagreen",
                )

                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )
            df_genre_rating = None

        with row3_2:
            st.subheader("Languages")
            st.markdown(
                f'<p style="background-color:#101010;color:#101010;font-size:24px;">--</p>',
                unsafe_allow_html=True,
            )

            df_lang = df[["film_name", "langs"]]
            df_lang = (
                df_lang[~(df_lang["langs"] == np.nan) & (df_lang["langs"].notna())]
                .reset_index()
                .drop("index", axis=1)
            )

            if isinstance(df_lang["langs"].values.tolist()[0], str):
                df_lang["langs"] = (
                    df_lang["langs"].dropna().apply(lambda x: ast.literal_eval(x))
                )

            df_lang["only_lang"] = df_lang["langs"]
            df_lang["only_lang"] = [
                [e for e in ele] if ele is not np.nan else ele
                for ele in df_lang["only_lang"].values.tolist()
            ]
            df_lang = df_lang["only_lang"].explode().reset_index()

            df_lang_cnt = (
                df_lang.groupby(by=["only_lang"])
                .count()
                .sort_values(by=["index"], ascending=False)
            )
            df_lang_cnt["only_lang"] = df_lang_cnt.index
            df_lang_cnt = df_lang_cnt.nlargest(10, "index")

            if len(df_lang_cnt) == 0:
                st.markdown("")

            else:
                fig = make_horizontal_bar_chart(
                    data=df_lang_cnt, y="only_lang", x="index", ccs="crimson"
                )

                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

            st.markdown(
                f'<p style="background-color:#101010;color:#101010;font-size:24px;">--</p>',
                unsafe_allow_html=True,
            )

            df_lang_rating = df[["rating", "langs"]]
            df_lang_rating = (
                df_lang_rating[
                    ~(df_lang_rating["langs"] == np.nan)
                    & (df_lang_rating["langs"].notna())
                ]
                .reset_index()
                .drop("index", axis=1)
            )

            if isinstance(df_lang_rating["langs"].values.tolist()[0], str):
                df_lang_rating["langs"] = df_lang_rating["langs"].apply(
                    lambda x: ast.literal_eval(x)
                )
            # else:
            df_lang_rating["only_lang"] = df_lang_rating[
                "langs"
            ]  # .apply(lambda x: list(ast.literal_eval(x)))

            df_lang_rating["only_lang"] = [
                [e for e in ele] if ele is not np.nan else ele
                for ele in df_lang_rating["only_lang"].values.tolist()
            ]
            df_lang_rating = (
                df_lang_rating[df_lang_rating["langs"].notna()]
                .reset_index()
                .drop("index", axis=1)
            )
            df_lang_rating.index = df_lang_rating["rating"]
            df_lang_rating = df_lang_rating["only_lang"].explode()

            df_lang_rating2 = pd.DataFrame(columns=["rating", "only_lang"])
            df_lang_rating2["rating"] = df_lang_rating.index.values.tolist()
            df_lang_rating2["only_lang"] = df_lang_rating.values.tolist()

            v = df_lang_rating2["only_lang"].value_counts()
            df_lang_rating2 = df_lang_rating2[
                df_lang_rating2["only_lang"].isin(v[v > 5].index.values.tolist())
            ]

            df_lang_rating_cnt = (
                df_lang_rating2.groupby(by=["only_lang"])
                .mean()
                .sort_values(by=["rating"], ascending=False)
            )
            df_lang_rating_cnt = df_lang_rating_cnt.nlargest(10, "rating")
            df_lang_rating_cnt["only_lang"] = df_lang_rating_cnt.index
            df_lang_rating_cnt.reset_index(drop=True, inplace=True)

            if len(df_lang_rating_cnt) == 0:
                st.markdown("#### Not enough data to show.")

            else:
                fig = make_horizontal_bar_chart(
                    data=df_lang_rating_cnt, y="only_lang", x="rating", ccs="crimson"
                )

                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )
            df_lang_rating = None

        with row3_3:
            st.subheader("Countries")
            # st.markdown(f'#### ----')
            st.markdown(
                f'<p style="background-color:#101010;color:#101010;font-size:24px;">--</p>',
                unsafe_allow_html=True,
            )

            df_country = df[["film_name", "countries"]]
            df_country = (
                df_country[
                    ~(df_country["countries"] == np.nan)
                    & (df_country["countries"].notna())
                ]
                .reset_index()
                .drop("index", axis=1)
            )

            if isinstance(df_country["countries"].values.tolist()[0], str):
                df_country["countries"] = df_country["countries"].apply(
                    lambda x: ast.literal_eval(x)
                )

            df_country["only_country"] = df_country["countries"]
            df_country["only_country"] = [
                [e for e in ele] if ele is not np.nan else ele
                for ele in df_country["only_country"].values.tolist()
            ]

            df_country = df_country["only_country"].explode().reset_index()
            df_country["index"] = df_country.index

            v = df_country["only_country"].value_counts()
            df_country = df_country[
                df_country["only_country"].isin(v[v > 5].index.values.tolist())
            ]

            df_country_cnt = (
                df_country.groupby(by=["only_country"])
                .count()
                .sort_values(by=["index"], ascending=False)
            )
            df_country_cnt["only_country"] = df_country_cnt.index
            # df_country_cnt=df_country_cnt.nlargest(10, 'index')

            if len(df_country_cnt) == 0:
                st.markdown("")

            else:
                fig = make_horizontal_bar_chart(
                    data=df_country_cnt.nlargest(10, "index"),
                    y="only_country",
                    x="index",
                    ccs="green",
                )

                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

            st.markdown(
                f'<p style="background-color:#101010;color:#101010;font-size:24px;">--</p>',
                unsafe_allow_html=True,
            )

            df_country_rating = df[["rating", "countries"]]
            df_country_rating = (
                df_country_rating[
                    ~(df_country_rating["countries"] == np.nan)
                    & (df_country_rating["countries"].notna())
                ]
                .reset_index()
                .drop("index", axis=1)
            )

            if isinstance(df_country_rating["countries"].values.tolist()[0], str):
                df_country_rating["countries"] = df_country_rating["countries"].apply(
                    lambda x: ast.literal_eval(x)
                )
            # else:
            df_country_rating["only_country"] = df_country_rating[
                "countries"
            ]  # .apply(lambda x: list(ast.literal_eval(x)))

            df_country_rating["only_country"] = [
                [e for e in ele] if ele is not np.nan else ele
                for ele in df_country_rating["only_country"].values.tolist()
            ]
            df_country_rating = (
                df_country_rating[df_country_rating["countries"].notna()]
                .reset_index()
                .drop("index", axis=1)
            )
            df_country_rating.index = df_country_rating["rating"]
            df_country_rating = df_country_rating["only_country"].explode()

            df_country_rating2 = pd.DataFrame(columns=["rating", "only_country"])
            df_country_rating2["rating"] = df_country_rating.index.values.tolist()
            df_country_rating2["only_country"] = df_country_rating.values.tolist()

            df_country_rating_cnt = (
                df_country_rating2.groupby(by=["only_country"])
                .mean()
                .sort_values(by=["rating"], ascending=False)
            )
            df_country_rating_cnt = df_country_rating_cnt.nlargest(10, "rating")
            df_country_rating_cnt["only_country"] = df_country_rating_cnt.index
            df_country_rating_cnt.reset_index(drop=True, inplace=True)

            if len(df_country_rating_cnt) == 0:
                st.markdown("#### Not enough data to show.")

            else:
                fig = make_horizontal_bar_chart(
                    data=df_country_rating_cnt,
                    y="only_country",
                    x="rating",
                    ccs="green",
                )

                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )
            df_country_rating = None

        st.write("")

        # top themes
        st.subheader("Themes")
        st.markdown(f"""###### Click on the theme to view your films""")

        df_theme = df[["rating", "themes"]]
        df_theme = (
            df_theme[~(df_theme["themes"] == np.nan) & (df_theme["themes"].notna())]
            .reset_index()
            .drop("index", axis=1)
        )

        if isinstance(df_theme["themes"].values.tolist()[0], str):
            df_theme["themes"] = (
                df_theme["themes"].dropna().apply(lambda x: ast.literal_eval(x))
            )

        df_theme["only_theme"] = df_theme["themes"]
        df_theme["only_theme_link"] = df_theme["themes"]

        df_theme["only_theme"] = [
            [e[0] for e in ele] if ele is not np.nan else ele
            for ele in df_theme["only_theme"].values.tolist()
        ]
        df_theme["only_theme_link"] = [
            [e[1] for e in ele] if ele is not np.nan else ele
            for ele in df_theme["only_theme_link"].values.tolist()
        ]
        df_theme = df_theme.drop("themes", axis=1)

        df_theme2 = (
            df_theme[["rating", "only_theme"]]
            .explode("only_theme")
            .reset_index(drop=True)
        )
        df_theme3 = df_theme["only_theme_link"].explode().reset_index()

        df_theme = df_theme2
        df_theme["only_theme_link"] = df_theme3["only_theme_link"]

        df_theme["index"] = df_theme.index

        df_theme["theme_and_link"] = (
            df_theme["only_theme"] + "||" + df_theme["only_theme_link"]
        )
        df_theme = df_theme.drop(["only_theme", "only_theme_link"], axis=1)

        df_theme_cnt = (
            df_theme.drop("rating", axis=1)
            .groupby(by=["theme_and_link"])
            .count()
            .sort_values(by=["index"], ascending=False)
        )

        df_theme_cnt = df_theme_cnt[
            ~df_theme_cnt.index.str.contains(
                "Show All", regex=False, case=False, na=False
            )
        ]
        df_theme_cnt["theme_and_link"] = df_theme_cnt.index
        df_theme_cnt.reset_index(drop=True, inplace=True)

        df_theme_cnt = df_theme_cnt.nlargest(10, "index")

        df_theme_cnt[["only_theme", "only_theme_link"]] = [
            val.split("||") for val in df_theme_cnt["theme_and_link"].values.tolist()
        ]
        df_theme_cnt = df_theme_cnt.drop("theme_and_link", axis=1)

        v = df_theme["theme_and_link"].value_counts()
        df_theme = df_theme[
            df_theme["theme_and_link"].isin(v[v > 5].index.values.tolist())
        ]
        df_theme_rating = (
            df_theme.drop("index", axis=1)
            .groupby(by=["theme_and_link"])
            .mean()
            .sort_values(by=["rating"], ascending=False)
        )

        df_theme_rating = df_theme_rating[
            ~df_theme_rating.index.str.contains(
                "Show All", regex=False, case=False, na=False
            )
        ]
        df_theme_rating["theme_and_link"] = df_theme_rating.index
        df_theme_rating.reset_index(drop=True, inplace=True)

        df_theme_rating = df_theme_rating.nlargest(10, "rating")
        df_theme_rating["rating"] = df_theme_rating["rating"].apply(
            lambda x: round(x, 1)
        )

        if len(df_theme_rating) == 0:
            df_theme_rating["only_theme"] = []
            df_theme_rating["only_theme_link"] = []
        else:
            df_theme_rating[["only_theme", "only_theme_link"]] = [
                val.split("||")
                for val in df_theme_rating["theme_and_link"].values.tolist()
            ]
        df_theme_rating = df_theme_rating.drop("theme_and_link", axis=1)

        df_theme = None

        st.markdown(f"""#### Type of films you **watched** the most...""")

        (
            row6_space1,
            row6_1,
            row6_space2,
            row6_2,
            row6_space3,
            row6_3,
            row6_space4,
        ) = st.columns((0.1, 2, 0.1, 40, 0.1, 2, 1.0))

        with row6_1:
            st.markdown("")
        with row6_2:
            if len(df_theme_cnt) == 0:
                st.markdown("#### Not enough data to show.")

            else:
                data = df_theme_cnt.to_dict(orient="list")
                idx = df_theme_cnt["only_theme"].tolist()

                root_link = f"https://letterboxd.com/{user_input}"

                source = ColumnDataSource(data=data)

                x_min = df_theme_cnt["index"].values.min() - 10
                x_min = 0 if x_min < 0 else x_min
                p = figure(
                    y_range=idx,
                    x_range=(x_min, df_theme_cnt["index"].values.max() + 5),
                    plot_height=350,
                    toolbar_location=None,
                    tools="tap",
                    background_fill_color="#101010",
                    border_fill_color="#101010",
                    outline_line_color="#101010",
                )
                p.grid.visible = False
                p.xaxis.visible = False
                p.yaxis.axis_line_width = 0
                p.yaxis.major_tick_line_width = 0
                p.yaxis.major_label_text_color = "lightgray"
                p.yaxis.major_label_text_font = "arial"
                p.yaxis.major_label_text_font_size = "19px"

                p.hbar(
                    y="only_theme",
                    right="index",
                    height=0.7,
                    source=source,
                    color="crimson",
                    hover_fill_color="#20b2aa",
                    hover_line_color="#20b2aa",
                )
                p.add_tools(HoverTool(tooltips="@index films"))

                url = root_link + "@only_theme_link"
                taptool = p.select(type=TapTool)
                taptool.callback = OpenURL(url=url)

                st.bokeh_chart(p, use_container_width=True)

        with row6_3:
            st.markdown("")

        st.markdown(f"""#### Types of films you *loved* the most...""")

        (
            row7_space1,
            row7_1,
            row7_space2,
            row7_2,
            row7_space3,
            row7_3,
            row7_space4,
        ) = st.columns((0.1, 2, 0.1, 40, 0.1, 2, 1.0))

        with row7_1:
            st.markdown("")
        with row7_2:
            if len(df_theme_rating) == 0:
                st.markdown("#### Not enough data to show.")

            else:

                data = df_theme_rating.to_dict(orient="list")
                idx = df_theme_rating["only_theme"].tolist()

                root_link = f"https://letterboxd.com/{user_input}"

                source = ColumnDataSource(data=data)

                x_min = df_theme_rating["rating"].values.min() - 10
                x_min = 0 if x_min < 0 else x_min
                p = figure(
                    y_range=idx,
                    x_range=(x_min, df_theme_rating["rating"].values.max() + 5),
                    plot_height=350,
                    toolbar_location=None,
                    tools="tap",
                    background_fill_color="#101010",
                    border_fill_color="#101010",
                    outline_line_color="#101010",
                )
                p.grid.visible = False
                p.xaxis.visible = False
                p.yaxis.axis_line_width = 0
                p.yaxis.major_tick_line_width = 0
                p.yaxis.major_label_text_color = "lightgray"
                p.yaxis.major_label_text_font = "arial"
                p.yaxis.major_label_text_font_size = "19px"

                p.hbar(
                    y="only_theme",
                    right="rating",
                    height=0.7,
                    source=source,
                    color="#20b2aa",
                    hover_fill_color="crimson",
                    hover_line_color="crimson",
                )
                p.add_tools(HoverTool(tooltips="@rating{0.0} rating"))

                url = root_link + "@only_theme_link"
                taptool = p.select(type=TapTool)
                taptool.callback = OpenURL(url=url)

                st.bokeh_chart(p, use_container_width=True)

        with row7_3:
            st.markdown("")

        # directors
        st.subheader("Some of your favourite directors and how you rated their films")

        marker_size_scale = 5
        director_df = df[["director", "avg_rating"]][
            ~(df["director"] == np.nan)
            & (df["director"].notna())
            & ~(df["director"] == "")
            & ~(df["director"] == " ")
        ]

        v = director_df["director"].value_counts()
        director_df_filtered = director_df[
            (director_df["director"].isin(v[v > 3].index.values.tolist()))
        ]
        if len(director_df_filtered) == 0:
            director_df_filtered = director_df
            marker_size_scale = 20
        director_df_rate = (
            director_df_filtered[["director", "avg_rating"]]
            .groupby(by="director")
            .mean()
            .reset_index()
            .sort_values(by=["avg_rating"], ascending=False)
        )
        director_df_count = (
            director_df_filtered[["director", "avg_rating"]]
            .groupby(by="director")
            .count()
            .sort_values(by=["avg_rating"], ascending=False)
            .reset_index()
            .rename(columns={"avg_rating": "count"})
        )
        director_df = (
            director_df_rate[["director", "avg_rating"]]
            .merge(director_df_count[["director", "count"]], on="director", how="inner")
            .sort_values(by=["avg_rating"], ascending=False)
            .nlargest(15, "count")
        )
        director_df["avg_rating"] = director_df["avg_rating"].map(lambda x: round(x, 2))

        if len(director_df) == 0:
            st.markdown(" ")

        else:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=director_df["avg_rating"].values.tolist(),
                        y=director_df["count"].values.tolist(),
                        text=director_df["count"].values.tolist(),
                        customdata=director_df["director"].values.tolist(),
                        mode="markers+text",
                        hovertemplate="<b>%{customdata}</b><br>"
                        + "%{x}<br>"
                        + "%{y} films<br><extra></extra>",
                        marker=dict(
                            size=np.interp(
                                director_df["count"].values,
                                (
                                    director_df["count"].values.min(),
                                    director_df["count"].values.max(),
                                ),
                                (30, 150),
                            ).tolist(),  # np.multiply(director_df['count'].values, 5).tolist(),
                            color=list(np.arange(100, 100 + (16 - 1) * 15, 15)),
                            showscale=False,
                            line=dict(width=0),
                        ),
                        texttemplate="<b>%{text}</b>",
                        textfont=dict(color="lightgray"),
                    )
                ]
            )
            fig.update(layout_coloraxis_showscale=False, layout_showlegend=False)
            fig.update_traces(showlegend=False)
            # fig.update_xaxes(visible=False)
            # fig.update_yaxes(visible=False)
            fig.layout.plot_bgcolor = "#101010"
            fig.update_layout(
                xaxis_title="Rating ⟶",
                yaxis_title="Number of titles ⟶",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
            )
            fig.layout.xaxis.titlefont.color = "lightgray"
            fig.layout.yaxis.titlefont.color = "lightgray"
            fig.layout.xaxis.showticklabels = False
            fig.layout.yaxis.showticklabels = False
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        st.markdown("""---""")

        # world map
        st.markdown("""## Look at the world through the films you've rated...""")

        df_country = df[["film_name", "countries"]]
        df_country = (
            df_country[
                ~(df_country["countries"] == np.nan) & (df_country["countries"].notna())
            ]
            .reset_index()
            .drop("index", axis=1)
        )

        if isinstance(df_country["countries"].values.tolist()[0], str):
            df_country["countries"] = df_country["countries"].apply(
                lambda x: ast.literal_eval(x)
            )

        df_country["only_country"] = df_country["countries"]
        df_country["only_country"] = [
            [e for e in ele] if ele is not np.nan else ele
            for ele in df_country["only_country"].values.tolist()
        ]

        df_country = df_country["only_country"].explode().reset_index()
        df_country["index"] = df_country.index

        df_country_cnt = (
            df_country.groupby(by=["only_country"])
            .count()
            .sort_values(by=["index"], ascending=False)
        )
        df_country_cnt["only_country"] = df_country_cnt.index

        df_country_cnt["COUNTRY"] = df_country_cnt["only_country"].astype(str)
        df_country_cnt = df_country_cnt.drop("only_country", axis=1)
        df_country_cnt["count"] = df_country_cnt["index"].astype(str)
        df_country_cnt = df_country_cnt.drop("index", axis=1)
        df_country_cnt.reset_index(drop=True, inplace=True)

        with open("./iso3_mapping.json") as f:
            country_mapping = json.load(f)

        df_country_cnt = df_country_cnt[
            df_country_cnt["COUNTRY"].isin(list(country_mapping.keys()))
        ]
        df_country_cnt["code"] = [
            country_mapping[cname]["code"]
            for cname in df_country_cnt["COUNTRY"].values.tolist()
            if cname in country_mapping.keys()
        ]

        if len(df_country_cnt) == 0:
            st.markdown(" ")

        else:
            fig = make_world_map(df_country_cnt)
            st.plotly_chart(fig, config={"displayModeBar": False})

        df_country_cnt = None

        st.markdown("""---""")

        st.markdown(
            '#### <h3><b> Happy watching & follow me on Letterboxd at\
              <a href=https://letterboxd.com/param_raval/ style="color: #b9babd;\
                text-decoration: underline;">param_raval</a>!</b></h5>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
