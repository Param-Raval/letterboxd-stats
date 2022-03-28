import plotly.express as px
# colorscales = px.colors.named_colorscales()
import plotly.graph_objects as go

from bokeh.models import ColumnDataSource, Grid, ImageURL, LinearAxis, Plot, Range1d
from bokeh.models import ColumnDataSource, OpenURL, TapTool, HoverTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.core.properties import value
from bokeh.models import Title

import pandas as pd

import textwrap
from datetime import datetime
from sorcery import dict_of


def make_horizontal_bar_chart(data, x, y, color=None, text=None, ccs=None):
    fig = px.bar(        
            data,
            y = y,
            x = x,
            orientation='h',
            color_discrete_sequence=[ccs]*len(data)
        )
    
    if "rating" in x:
        fig.update_traces(hovertemplate=
                          '%{x:.2f} <b>rating</b>')
    else:
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

def make_world_map(df_country_cnt):
    fig = go.Figure(data=go.Choropleth(
                locations = df_country_cnt['code'],
                # locationmode="country names",
                z = df_country_cnt['count'],
                text=df_country_cnt['COUNTRY'],
                hovertemplate='<b>%{text}</b><br>%{z} films<extra></extra>',
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
            # scope='world'
        ),
        width=1400,
        height=700
        )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(showlegend=False)
    fig.update(layout_showlegend=False)
    fig.update_traces(showscale=False)

    return fig
            
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


def get_image_from_url(image_url, film_row):
  source = ColumnDataSource(dict(
      url = [image_url],
      x1  = [0],
      y1  = [0],
      w1  = [230],
      h1  = [345],
      x2  = [0],
      y2  = [0],
  ))

  plot = Plot(
      title=None, width=230, height=345,
      min_border=0, toolbar_location=None,
      background_fill_color = "#101010",
      border_fill_color = "#101010",
      outline_line_color=  "#101010")

  image1 = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
  plot.add_glyph(source, image1)
  subtitle_string = f"{film_row['film_name']}, {int(film_row['year'])} || {inverse_transform_ratings(str(film_row['rating']))}"
  wrapper = textwrap.TextWrapper(width=30)
  subtitle_string = wrapper.fill(text=subtitle_string)
  subtitle_string = subtitle_string.split('||')[0].strip('\n') + '\n' + subtitle_string.split('||')[1].replace('\n', '')

  plot.add_layout(Title(text=subtitle_string, 
                        align="center", text_color="white"), "below")

  return plot


def get_meta_dict(ratings_data):
    this_month_name = datetime.strptime(f"{datetime.today().month}", "%m").strftime('%B')
    this_month = f"{this_month_name},{datetime.today().year}"
    this_year = f"{datetime.today().year}"
    last_year = f"{int(this_year)-1}"

    ratings_data['rated_date_year']=ratings_data['rated_date'].map(lambda x: f"{x.year}")
    ratings_data['rated_date_month']=ratings_data['rated_date'].map(lambda x: datetime.strptime(f"{x.month}", "%m").strftime('%B'))
    ratings_data['rated_date_time_day']=ratings_data['rated_date'].map(lambda x: "Morning" if x.hour<16 else("Evening" if x.hour >= 16 and x.hour<20 else "Night"))

    nfilms_month_wise = ratings_data['rating'].groupby(ratings_data['rated_date'].map(lambda x: str(datetime.strptime(f"{x.month}", "%m").strftime('%B'))+f",{x.year}")).count().reset_index()
    nfilms_this_month = nfilms_month_wise[nfilms_month_wise['rated_date']==this_month]['rating'].values.tolist()
    nfilms_this_month = nfilms_this_month[0] if len(nfilms_this_month)>0 else 0

    nfilms_year_wise = ratings_data['rating'].groupby(ratings_data['rated_date'].map(lambda x: f"{x.year}")).count().reset_index()
    nfilms_this_year = nfilms_year_wise[nfilms_year_wise['rated_date']==this_year]['rating'].values.tolist()
    nfilms_this_year = nfilms_this_year[0] if len(nfilms_this_year)>0 else 0

    nfilms_most_year, nfilms_most_year_count = nfilms_year_wise.sort_values(by=['rating']).values.tolist()[-1]

    last_year_month_stats = ratings_data['film_name'][ratings_data['rated_date_year']==last_year].groupby(ratings_data['rated_date_month']).count().reset_index().sort_values(by=['film_name']).values.tolist()
    
    nfilms_last_year_most_month, nfilms_last_year_most_month_count = last_year_month_stats[-1] if len(last_year_month_stats)>0 else ("None", 0)

    last_year_month_stats = ratings_data['rating'][ratings_data['rated_date_year']==last_year].groupby(ratings_data['rated_date_month']).mean().reset_index().sort_values(by=['rating']).values.tolist()
    nfilms_last_year_most_rated_month, nfilms_last_year_most_rated_month_val = last_year_month_stats[-1] if len(last_year_month_stats)>0 else ("None", 0)
    nfilms_last_year_most_rated_month_val=round(nfilms_last_year_most_rated_month_val,1)

    nfilms_time_of_day = ratings_data['film_name'].groupby(ratings_data['rated_date_time_day']).count().reset_index().sort_values(by=['film_name'])['rated_date_time_day'].values[-1]

    first_film_name, first_film_rating = ratings_data.sort_values(by=['rated_date'])[['film_name', 'rating']].values.tolist()[0]
    latest_film_name, latest_film_rating = ratings_data.sort_values(by=['rated_date'])[['film_name', 'rating']].values.tolist()[-1]

    d = dict_of(nfilms_this_month, nfilms_this_year, nfilms_last_year_most_month,
                nfilms_last_year_most_month_count,
                nfilms_last_year_most_rated_month,
                nfilms_last_year_most_rated_month_val,
                nfilms_time_of_day,
                first_film_name, first_film_rating,
                latest_film_name, latest_film_rating,
                nfilms_most_year, nfilms_most_year_count)
    ratings_data=ratings_data.drop('rated_date_year', axis=1)
    ratings_data=ratings_data.drop('rated_date_month', axis=1)
    ratings_data=ratings_data.drop('rated_date_time_day', axis=1)
    return d
