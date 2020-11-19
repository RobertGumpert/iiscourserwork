import pandas


def add_region_if_not_exist(map_year_and_data_frame):
    country_and_region = dict()
    update_data_frames = []
    for year, data_frame in map_year_and_data_frame.items():
        if map_year_and_data_frame[year]['Region'].count() == 0:
            update_data_frames.append(year)
            continue
        country_and_region.update(dict(zip(data_frame['Country'], data_frame['Region'])))
    for data_frame in update_data_frames:
        map_year_and_data_frame[data_frame].loc[
            map_year_and_data_frame[data_frame]['Country'].isin(country_and_region.keys()), 'Region'] = \
            map_year_and_data_frame[data_frame]['Country'].map(country_and_region)
    return country_and_region


def concat(map_year_and_data_frame):
    count = 0
    for year, data_frame in map_year_and_data_frame.items():
        count += len(data_frame.index)
    concat_data_frame = pandas.concat(list(map_year_and_data_frame.values()))
    params = ['GDP', 'Family', 'Health', 'Freedom', 'Corruption']
    max_value = concat_data_frame[params].values.max()
    for param in params:
        concat_data_frame[param] = [
            (x / max_value) * 100 for x in concat_data_frame[param]
        ]
    return concat_data_frame


# 'region' => <row_data_frame>
def group_by_region(concat_data_frame):
    map_region_and_rows = dict()
    for region in concat_data_frame['Region'].unique():
        map_region_and_rows[region] = concat_data_frame.loc[concat_data_frame['Region'] == region]
    return map_region_and_rows


# 'country' => <row_data_frame>
def group_by_country(concat_data_frame):
    map_country_and_rows = dict()
    for country in concat_data_frame['Country'].unique():
        map_country_and_rows[country] = concat_data_frame.loc[concat_data_frame['Country'] == country]
    return map_country_and_rows
