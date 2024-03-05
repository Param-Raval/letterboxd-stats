import webbrowser
from datetime import datetime

from sorcery import dict_of


def get_meta_dict(ratings_data):
    this_month_name = datetime.strptime(f"{datetime.today().month}", "%m").strftime(
        "%B"
    )
    this_month = f"{this_month_name},{datetime.today().year}"
    this_year = f"{datetime.today().year}"
    last_year = f"{int(this_year)-1}"

    ratings_data["rated_date_year"] = ratings_data["rated_date"].map(
        lambda x: f"{x.year}"
    )
    ratings_data["rated_date_month"] = ratings_data["rated_date"].map(
        lambda x: datetime.strptime(f"{x.month}", "%m").strftime("%B")
    )
    ratings_data["rated_date_time_day"] = ratings_data["rated_date"].map(
        lambda x: "Morning"
        if x.hour < 16
        else ("Evening" if x.hour >= 16 and x.hour < 20 else "Night")
    )

    nfilms_month_wise = (
        ratings_data["rating"]
        .groupby(
            ratings_data["rated_date"].map(
                lambda x: str(datetime.strptime(f"{x.month}", "%m").strftime("%B"))
                + f",{x.year}"
            )
        )
        .count()
        .reset_index()
    )
    nfilms_this_month = nfilms_month_wise[
        nfilms_month_wise["rated_date"] == this_month
    ]["rating"].values.tolist()
    nfilms_this_month = nfilms_this_month[0] if len(nfilms_this_month) > 0 else 0

    nfilms_year_wise = (
        ratings_data["rating"]
        .groupby(ratings_data["rated_date"].map(lambda x: f"{x.year}"))
        .count()
        .reset_index()
    )
    nfilms_this_year = nfilms_year_wise[nfilms_year_wise["rated_date"] == this_year][
        "rating"
    ].values.tolist()
    nfilms_this_year = nfilms_this_year[0] if len(nfilms_this_year) > 0 else 0

    nfilms_most_year, nfilms_most_year_count = nfilms_year_wise.sort_values(
        by=["rating"]
    ).values.tolist()[-1]

    last_year_month_stats = (
        ratings_data["film_name"][ratings_data["rated_date_year"] == last_year]
        .groupby(ratings_data["rated_date_month"])
        .count()
        .reset_index()
        .sort_values(by=["film_name"])
        .values.tolist()
    )

    nfilms_last_year_most_month, nfilms_last_year_most_month_count = (
        last_year_month_stats[-1] if len(last_year_month_stats) > 0 else ("None", 0)
    )

    last_year_month_stats = (
        ratings_data["rating"][ratings_data["rated_date_year"] == last_year]
        .groupby(ratings_data["rated_date_month"])
        .mean()
        .reset_index()
        .sort_values(by=["rating"])
        .values.tolist()
    )
    nfilms_last_year_most_rated_month, nfilms_last_year_most_rated_month_val = (
        last_year_month_stats[-1] if len(last_year_month_stats) > 0 else ("None", 0)
    )
    nfilms_last_year_most_rated_month_val = round(
        nfilms_last_year_most_rated_month_val, 1
    )

    nfilms_time_of_day = (
        ratings_data["film_name"]
        .groupby(ratings_data["rated_date_time_day"])
        .count()
        .reset_index()
        .sort_values(by=["film_name"])["rated_date_time_day"]
        .values[-1]
    )

    first_film_name, first_film_rating = ratings_data.sort_values(by=["rated_date"])[
        ["film_name", "rating"]
    ].values.tolist()[0]
    latest_film_name, latest_film_rating = ratings_data.sort_values(by=["rated_date"])[
        ["film_name", "rating"]
    ].values.tolist()[-1]

    d = dict_of(
        nfilms_this_month,
        nfilms_this_year,
        nfilms_last_year_most_month,
        nfilms_last_year_most_month_count,
        nfilms_last_year_most_rated_month,
        nfilms_last_year_most_rated_month_val,
        nfilms_time_of_day,
        first_film_name,
        first_film_rating,
        latest_film_name,
        latest_film_rating,
        nfilms_most_year,
        nfilms_most_year_count,
    )
    ratings_data = ratings_data.drop("rated_date_year", axis=1)
    ratings_data = ratings_data.drop("rated_date_month", axis=1)
    ratings_data = ratings_data.drop("rated_date_time_day", axis=1)
    return d


def open_link(url):
    webbrowser.open_new_tab(url)
