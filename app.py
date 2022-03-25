import streamlit as st
st.set_page_config(layout="wide")

import plotly.express as px
# colorscales = px.colors.named_colorscales()
import pandas as pd
import random
import ast
import numpy as np
import plotly.graph_objects as go

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from pprint import pprint
from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
import time

_domain = "https://letterboxd.com"
film_ratings = []
film_cache = pd.read_excel('./film_cache.xlsx')

def get_film_data(filmget):
  # film_page="https://letterboxd.com/film/phantom-thread/"
  # filmget = requests.get(film_page)
  # session = requests.Session()
  # filmget = session.get(film_page)

  film_soup = BeautifulSoup(filmget.content, 'html.parser')
  
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

def inverse_transform_ratings(some_number):
    stars = {
            '1.0':"★",
            '2.0':"★★",
            '3.0':"★★★",
            '4.0':"★★★★",
            '5.0':"★★★★★",
            '0.5':"½",
            '1.5':"★½",
            '2.5':"★★½",
            '3.5':"★★★½",
            '4.5':"★★★★½"
        }
    try:
        return stars[some_number]
    except:
        return -1


def get_ratings_from_futures(ratings_page):
  # check to see page was downloaded correctly
  if ratings_page.status_code != 200:
      print("PAGE NOT FOUND")
      return None

  soup = BeautifulSoup(ratings_page.content, 'html.parser')
  # browser.get(following_url)

  # grab the main film grid
  table = soup.find('ul', class_='poster-list')
  if table is None:
      return None

  films = table.find_all('li')
  film_ratings = list()

  # iterate through friends
  for film in films:
      panel = film.find('div').find('img')
      film_name = panel['alt']
      stars = transform_ratings(film.find('p', class_='poster-viewingdata').get_text().strip())
      film_link = _domain + film.find('div').get('data-target-link')
      if stars == -1:
          continue
      film_ratings.append((film_name, stars, film_link))

  return film_ratings


def get_film_df(user_name):
    
    rlink = f"https://letterboxd.com/{user_name}/films/ratings/"
    ratings_page = requests.get(rlink)

    if ratings_page.status_code != 200:
        print("PAGE NOT FOUND")
        return 0

    soup = BeautifulSoup(ratings_page.content, 'html.parser')
    try:
        num_pages=int(soup.find('div', class_="pagination").find_all('a')[-1].contents[0])
        url_list = [rlink] + [rlink+f"page/{idx}/" for idx in range(2,num_pages+1)]
    except:
        url_list = [rlink]

    with FuturesSession() as session:
      t1=time.time()
      futures = [session.get(ele) for ele in tqdm(url_list)]
      ratings = []
      for future in as_completed(futures):
          temp = get_ratings_from_futures(future.result())
          if temp is None:
            return 0
          ratings = ratings + temp

      print(len(ratings), time.time()-t1)

    st.markdown(f"#### 2. Fetched {len(ratings)} ratings from {user_name}! Fetching movie data...")
            
    films_url_list = [ele[-1] for ele in ratings]
    films_url_list_new = [film_url for film_url in films_url_list if film_url not in film_cache['lbxd_link'].values.tolist()]
    print(len(films_url_list_new))

    final_film_data=[]
    batch_size=30

    with FuturesSession() as session:
      t1=time.time()
      futures = [session.get(ele) for ele in tqdm(films_url_list_new)]
      ffd = [get_film_data(future.result()) for future in as_completed(futures)]

      print(len(ffd), time.time()-t1)

    st.markdown(f"#### 3. Fetched {len(ratings)} movies!")
    print(f"Fetched {len(ffd)} movies! {len(films_url_list)-len(films_url_list_new)} found in cache.")

    #all ratings of the user
    ratings_df = pd.DataFrame(ratings, columns=['film_name', 'rating', 'lbxd_link'])

    #only newly fetched film data
    column_list = ['lbxd_link', 'year', 'director', 'avg_rating', 'countries', 'langs', 'genres', 'themes', 'top_cast']
    film_df = pd.DataFrame(ffd, columns=column_list)
    
    #ratings and film data of newly fetched movies
    # df = ratings_df.join(film_df.set_index('lbxd_link'), on='lbxd_link')

    #ratings only of newly fetched movies
    ratings_df_new = ratings_df[ratings_df['lbxd_link'].isin(films_url_list_new)]
    df_new = ratings_df_new.join(film_df.set_index('lbxd_link'), on='lbxd_link')
    new_film_cache = pd.concat([df_new.drop('rating', axis=1), film_cache], ignore_index=True)
    new_film_cache.to_excel('./film_cache.xlsx', index=False)
    new_film_cache=None

    #film cache of user's films
    int_film_cache = film_cache[film_cache['lbxd_link'].isin([film_url for film_url in films_url_list if film_url in film_cache['lbxd_link'].values.tolist()])]

    #add user's ratings to existing film cache
    ratings_df_film_cache = ratings_df.join(int_film_cache.drop('film_name', axis=1).set_index('lbxd_link'), on='lbxd_link', how='inner')
    
    ratings_df_film_cache['genres'] = ratings_df_film_cache['genres'].apply(lambda x: ast.literal_eval(x) if x!="nan" and x is not np.nan else np.nan)
    ratings_df_film_cache['themes'] = ratings_df_film_cache['themes'].apply(lambda x: ast.literal_eval(x) if x!="nan" and x is not np.nan else np.nan)
    ratings_df_film_cache['langs'] = ratings_df_film_cache['langs'].apply(lambda x: ast.literal_eval(x) if x!="nan" and x is not np.nan else np.nan)
    ratings_df_film_cache['countries'] = ratings_df_film_cache['countries'].apply(lambda x: ast.literal_eval(x) if x!="nan" and x is not np.nan else np.nan)
    
    #add new ratings+film data to existing ratings+film cache
    final_df_to_use = pd.concat([df_new, ratings_df_film_cache], ignore_index=True)
    return final_df_to_use


def make_horizontal_bar_chart(data, x, y, color=None, text=None, ccs=None):
    fig = px.bar(        
            data,
            y = y,
            x = x,
            orientation='h',
            color_discrete_sequence=[ccs]*len(data)
        )
    fig.update_traces(hovertemplate=
                      '%{x} <b>films</b>')
    fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    fig.update_layout(yaxis_visible=True, yaxis_showticklabels=True, xaxis_title=None, 
                    yaxis_title=None, xaxis_visible=False, plot_bgcolor='#101010')
    fig.update_traces(showlegend=False,
        marker_coloraxis=None
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(yaxis = dict(ticks="outside", tickcolor="#101010", ticklen=10, tickfont = dict(size=20)))
    return fig

def make_bar_chart(data, x, y, color=None, text=None, ccs=None):
    fig = px.bar(        
                data,
                x = x,
                y = y,
                color = color,
                text=text,
                color_continuous_scale=ccs
            )
    if "rating" in y:
        fig.update_traces(hovertemplate=
                          '%{y:.2f} <b>Rating</b>' +
                          '<br>%{x}<br>')
    else:
        fig.update_traces(hovertemplate=
                          '%{y} <b>Films</b>' +
                          '<br>%{x}<br>')
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, xaxis_title=None, plot_bgcolor='#101010')
    fig.update(layout_coloraxis_showscale=False)
    return fig


def main():
    #Main page
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

    row0_1.title('Analyzing Your Letterboxd Profile')

    with row0_2:
        st.write('')

    row0_2.subheader(
        'A Streamlit web app by Param Raval')

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

    with row1_1:
        st.markdown(
            "**To begin, please enter your [Letterboxd profile](https://www.letterboxd.com/) (or just use mine!).** 👇")

    row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row2_1:
        default_username = "param_raval"
        user_input = st.text_input(
            "Input your own Letterboxd profile name (e.g. param_raval)", "param_raval")
        need_help = st.expander('Need help? 👉')
        with need_help:
            st.markdown(
                "Having trouble finding your Letterboxd profile? Head to the [Letterboxd website](https://www.letterboxd.com/) and click profile in the top right corner.")

        if not user_input:
            user_input = f"{default_username}"

        bt1 = st.button("Go!")
    
    if not bt1:
        st.markdown(f"")
    else:
        if user_input==default_username:
            df = pd.read_excel('./letterboxd_film_data1.xlsx')
        else:
            st.markdown(f"#### 1. Fetching profile of {user_input}")
            df = get_film_df(user_input)
            if isinstance(df, int) and df==0:
                st.markdown(f"### Username {user_input} not found!")
                return 0

        st.markdown(f"### Analyzing Profile of {user_input}")

        st.markdown(f"#### Number of films by release year")
        
        df_count = df[['film_name', 'year']].groupby(by=['year']).count()
        df_count['year'] = df_count.index
        df_count['decade'] =  df_count.apply(lambda row: row.year - row.year%10, axis=1).astype(int)
        df_count['year_rating'] = df[['rating', 'year']].groupby(by=['year']).mean()['rating']

        df['decade'] =  df.apply(lambda row: row.year - row.year%10, axis=1).astype(int)

        df_count2 = df[['film_name', 'decade']].groupby(by=['decade']).count()
        df_count2['decade'] = df_count2.index
        df_count2['decade_rating'] = df_count[['year_rating', 'decade']].groupby(by=['decade']).mean()['year_rating']

        fig = make_bar_chart( data = df_count,
                        x = "year",
                        y = "film_name",
                        color = "year",
                        ccs="sunset")
        st.plotly_chart(fig, use_container_width=True)



        st.markdown(f"#### Number of films by release decade")

        fig = make_bar_chart( data = df_count2,
                        x = "decade",
                        y = "film_name",
                        color = "decade",
                        ccs="emrld")
        st.plotly_chart(fig, use_container_width=True)


        st.markdown(f"#### Average rating by release decade")
        
        fig = make_bar_chart( data = df_count2,
                        x = "decade",
                        y = "decade_rating",
                        color = "decade",
                        ccs="agsunset")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"#### Average rating by release year")
        
        fig = make_bar_chart( data = df_count,
                        x = "year",
                        y = "year_rating",
                        color = "year",
                        ccs="viridis")
        st.plotly_chart(fig, use_container_width=True)


        st.markdown(f"#### Highest rated decades")
        hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        
        for name, group in df_count[['year', 'decade']][df_count['decade'].isin(df_count2.nlargest(3, 'decade_rating')['decade'].values.tolist())].groupby(by=['decade']):
            need_help = st.expander(f'{name}s')
            with need_help:
                df_group = df[['film_name', 'year', 'rating', 'lbxd_link']][df['year'].isin(group['year'].values.tolist())].nlargest(10, 'rating')
                # st.dataframe(df_group)

                for idx, row in df_group.iterrows():
                    st.markdown(f"""* [{row['film_name']}]({row['lbxd_link']}) ({row['year']}) {inverse_transform_ratings(str(row['rating']))}""")
                
        st.write('')
        row3_space1, row3_1, row3_space2, row3_2, row3_space3, row3_3, row4_space4 = st.columns(
            (.1, 15, .1, 15, .1, 15, 1.))
        with row3_1:
            st.subheader('Genres')

            df_genre = df[['film_name', 'genres']]
            df_genre=df_genre[~(df_genre['genres']==np.nan) & (df_genre['genres'].notna())].reset_index().drop('index', axis=1)

            if isinstance(df_genre['genres'].values.tolist()[0], str):
              df_genre['genres'] = df_genre['genres'].apply(lambda x: ast.literal_eval(x))
            # else:
            df_genre['only_genre'] = df_genre['genres'] #.apply(lambda x: list(ast.literal_eval(x)))

            df_genre['only_genre'] = [[e[0] for e in ele] if ele is not np.nan else ele for ele in df_genre['only_genre'].values.tolist()]
            df_genre=df_genre[df_genre['genres'].notna()].reset_index().drop('index', axis=1)
            df_genre=df_genre['only_genre'].explode().reset_index()
            df_genre['index'] = df_genre.index

            df_genre_cnt=df_genre.groupby(by=['only_genre']).count().sort_values(by=['index'], ascending=False)
            df_genre_cnt['only_genre'] = df_genre_cnt.index
            df_genre_cnt=df_genre_cnt.nlargest(10, 'index')

            fig = make_horizontal_bar_chart(data = df_genre_cnt,
                                      y = "only_genre",
                                      x = "index",
                                      ccs="lightseagreen")
            
            st.plotly_chart(fig, use_container_width=True)

        with row3_2:
            st.subheader('Languages')
            
            df_lang = df[['film_name', 'langs']]
            df_lang=df_lang[~(df_lang['langs']==np.nan) & (df_lang['langs'].notna())].reset_index().drop('index', axis=1)

            if isinstance(df_lang['langs'].values.tolist()[0], str):
                df_lang['langs'] = df_lang['langs'].dropna().apply(lambda x: ast.literal_eval(x))
            
            df_lang['only_lang'] = df_lang['langs']
            df_lang['only_lang'] = [[e for e in ele] if ele is not np.nan else ele for ele in df_lang['only_lang'].values.tolist()]
            df_lang=df_lang['only_lang'].explode().reset_index()

            df_lang_cnt=df_lang.groupby(by=['only_lang']).count().sort_values(by=['index'], ascending=False)
            df_lang_cnt['only_lang'] = df_lang_cnt.index
            df_lang_cnt=df_lang_cnt.nlargest(10, 'index')

            fig = make_horizontal_bar_chart(data = df_lang_cnt,
                                      y = "only_lang",
                                      x = "index",
                                      ccs="crimson")

            st.plotly_chart(fig, use_container_width=True)

        with row3_3:
            st.subheader('Countries')

            
            df_country = df[['film_name', 'countries']]
            df_country=df_country[~(df_country['countries']==np.nan) & (df_country['countries'].notna())].reset_index().drop('index', axis=1)

            if isinstance(df_country['countries'].values.tolist()[0], str):
              df_country['countries'] = df_country['countries'].apply(lambda x: ast.literal_eval(x))

            df_country['only_country'] = df_country['countries']
            df_country['only_country'] = [[e for e in ele] if ele is not np.nan else ele for ele in df_country['only_country'].values.tolist()]

            df_country=df_country['only_country'].explode().reset_index()
            df_country['index'] = df_country.index

            df_country_cnt=df_country.groupby(by=['only_country']).count().sort_values(by=['index'], ascending=False)
            df_country_cnt['only_country'] = df_country_cnt.index
            df_country_cnt=df_country_cnt.nlargest(10, 'index')

            fig = make_horizontal_bar_chart(data = df_country_cnt,
                                      y = "only_country",
                                      x = "index",
                                      ccs="crimson")
            
            st.plotly_chart(fig, use_container_width=True)

        st.write('')

        #top themes
        st.subheader('Top Themes')

        df_theme = df[['film_name', 'themes']]
        df_theme=df_theme[~(df_theme['themes']==np.nan) & (df_theme['themes'].notna())].reset_index().drop('index', axis=1)

        if isinstance(df_theme['themes'].values.tolist()[0], str):
            df_theme['themes'] = df_theme['themes'].dropna().apply(lambda x: ast.literal_eval(x))
        
        df_theme['only_theme'] = df_theme['themes']
        df_theme['only_theme'] = [[e[0] for e in ele] if ele is not np.nan else ele for ele in df_theme['only_theme'].values.tolist()]
        df_theme=df_theme['only_theme'].explode().reset_index()
        df_theme['index'] = df_theme.index

        df_theme_cnt=df_theme.groupby(by=['only_theme']).count().sort_values(by=['index'], ascending=False)
        
        df_theme_cnt=df_theme_cnt[~df_theme_cnt.index.str.contains('Show All', regex=False, case=False, na=False)]
        df_theme_cnt['only_theme'] = df_theme_cnt.index
        df_theme_cnt=df_theme_cnt.nlargest(10, 'index')

        fig = make_horizontal_bar_chart(data = df_theme_cnt,
                                      y = "only_theme",
                                      x = "index",
                                      text='index',
                                      ccs="lightseagreen")
        st.plotly_chart(fig, use_container_width=True)

        # fig = px.bar(        
        #         df_theme_cnt,
        #         y = "only_theme",
        #         x = "index",
        #         text='index',
        #         # color = "only_genre",
        #         # color_continuous_scale="emrld",
        #         orientation='h',
        #         color_discrete_sequence=["lightseagreen"]*len(df_theme_cnt)
        #     )
        # fig.update_traces(hovertemplate=
        #                   '%{x} <b>films</b>')
        # fig.update_layout(yaxis_visible=True, yaxis_showticklabels=True, xaxis_title=None,
        #                  yaxis_title=None, xaxis_visible=False, plot_bgcolor='#101010')
        # fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
        # fig.update(layout_coloraxis_showscale=False)
        # fig.update_layout(yaxis = dict(ticks="outside", tickcolor="#101010", ticklen=10, tickfont = dict(size=20)))
        # st.plotly_chart(fig, use_container_width=True)
    
        #world map
        # row5_space1, row5_1, row5_space2 = st.columns(
        #     (.1, 50, .1))
        
        # with row5_1:
        st.markdown("""### Your world in films""")
        df_country_cnt2 = df_country_cnt
        df_country_cnt2['COUNTRY'] = df_country_cnt2['only_country'].astype(str)
        df_country_cnt2 = df_country_cnt2.drop('only_country', axis=1)
        df_country_cnt2['count'] = df_country_cnt2['index'].astype(str)
        df_country_cnt2 = df_country_cnt2.drop('index', axis=1)
        df_country_cnt2.reset_index(drop=True, inplace=True)
        
        fig = go.Figure(data=go.Choropleth(
            locations = df_country_cnt2['COUNTRY'],
            locationmode="country names",
            z = df_country_cnt2['count'],
            colorscale = 'greens',
            autocolorscale=False,
            reversescale=True,
            marker_line_color='darkslategray',
            marker_line_width=0.5))

        fig.update_layout(
            geo=dict(
                bgcolor='#101010',
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular',
                scope='world'
            ),
            width=1400,
            height=700
            )
        fig.update(layout_coloraxis_showscale=False)
        fig.update_traces(showlegend=False)
        fig.update(layout_showlegend=False)
        fig.update_traces(showscale=False)
        st.plotly_chart(fig)
        #top nanogenres
        # st.subheader('Top Nanogenres')

        # df_ng = df[['film_name', 'nanogenres']]
        # df_ng['only_ng'] = df_ng['nanogenres'].dropna().apply(lambda x: ast.literal_eval(x))
        # df_ng['only_ng'] = [[e[0] for e in ele] if ele is not np.nan else ele for ele in df_ng['only_ng'].values.tolist()]
        # df_ng=df_ng[df_ng['nanogenres'].notna()].reset_index().drop('index', axis=1)
        # df_ng=df_ng.set_index(['film_name', 'nanogenres']).apply(lambda x: x.explode()).reset_index()

        # df_ng_cnt=df_ng.groupby(by=['only_ng']).count().sort_values(by=['film_name'], ascending=False).drop('nanogenres', axis=1)
        # # df_ng_cnt = df_ng_cnt[~df_ng_cnt.index.isin(['Show All…'])]

        # # df_ng_cnt = df_ng_cnt[~df_ng_cnt.index.isin(['Show All…'])]
        # df_ng_cnt=df_ng_cnt[~df_ng_cnt.index.str.contains('Show All', regex=False, case=False, na=False)]
        # df_ng_cnt['only_ng'] = df_ng_cnt.index
        # df_ng_cnt=df_ng_cnt.nlargest(10, 'film_name')

        # fig = px.bar(        
        #         df_ng_cnt,
        #         y = "only_ng",
        #         x = "film_name",
        #         text='film_name',
        #         # color = "only_genre",
        #         # color_continuous_scale="emrld",
        #         orientation='h',
        #         color_discrete_sequence=["lightseagreen"]*len(df_ng_cnt)
        #     )
        # fig.update_traces(hovertemplate=
        #                   '%{x} <b>films</b>')
        # fig.update_layout(yaxis_visible=True, yaxis_showticklabels=True, xaxis_title=None,
        #              yaxis_title=None, xaxis_visible=False, plot_bgcolor='#101010')
        # fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
        # fig.update(layout_coloraxis_showscale=False)
        # fig.update_layout(yaxis = dict(ticks="outside", tickcolor="#101010", ticklen=10, tickfont = dict(size=20)))
        # st.plotly_chart(fig, use_container_width=True)
                

if __name__ == "__main__":
    main()