import numpy as np
from fontTools.misc.cython import returns
from matplotlib import pyplot as plt

def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    # Check if we are filtering by overall.
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]
    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                      ascending=False).reset_index()

    # Group by 'region' and sum the medal counts.
    # x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']]

    # Calculate the total number of medals.
    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']
    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')
    # Sort by the 'Gold' column in descending order.
    # x = x.sort_values('Gold', ascending=False)

    return x

def medal_tally(df):
    medal_tally = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    medal_tally.loc[:, ~medal_tally.columns.duplicated()].groupby('NOC', as_index=False)[
        ['Gold', 'Silver', 'Bronze']].sum().sort_values('Gold', ascending=False)
    medal_tally = medal_tally.loc[:, ~medal_tally.columns.duplicated()]
    medal_tally['total'] = medal_tally[['Gold', 'Silver', 'Bronze']].sum(axis=1)
    medal_tally = medal_tally.groupby('NOC', as_index=False)[['Gold', 'Silver', 'Bronze', 'total']].sum().sort_values(
        'Gold', ascending=False)

    medal_tally['Gold'] = medal_tally['Gold'].astype('int')
    medal_tally['Silver'] = medal_tally['Silver'].astype('int')
    medal_tally['Bronze'] = medal_tally['Bronze'].astype('int')
    medal_tally['total'] = medal_tally['total'].astype('int')

    return medal_tally

def conuntry_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    # country = df['region'].unique().tolist()
    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years, country

def participating_nations_over_time(df):
    # Sample DataFrame (Replace this with your actual DataFrame)
    nation_over_time = df.drop_duplicates(['Year', 'region'])['Year'].value_counts().reset_index()

    # Fix column names
    nation_over_time.columns = ['Year', 'No of Countries']

    # Ensure 'Year' column is sorted
    nation_over_time = nation_over_time.sort_values('Year')
    return nation_over_time

def participating_nations_over_time_1(df):
    # Check for duplicate columns and remove if needed
    # nation_over_time = nation_over_time.loc[:, ~nation_over_time.columns.duplicated()]
    events_over_time = df.drop_duplicates(['Year', 'Event'])['Year'].value_counts().reset_index()

    # Fix column names
    events_over_time.columns = ['Year', 'Event']

    # Ensure 'Year' column is sorted
    events_over_time = events_over_time.sort_values('Year')

    return events_over_time

def participating_nations_over_time_2(df):
    # Check for duplicate columns and remove if needed
    # nation_over_time = nation_over_time.loc[:, ~nation_over_time.columns.duplicated()]
    atheletes_over_time = df.drop_duplicates(['Year', 'Name'])['Year'].value_counts().reset_index()

    # Fix column names
    atheletes_over_time.columns = ['Year', 'Name']

    # Ensure 'Year' column is sorted
    atheletes_over_time = atheletes_over_time.sort_values('Year')

    return atheletes_over_time


def most_successful(df, sport):
    temp_df = df.dropna(subset=['Medal'])  # Remove rows where Medal is NaN

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # Count medals per athlete and reset index for merging
    top_athletes = temp_df['Name'].value_counts().reset_index()
    top_athletes.columns = ['index', 'Name_x']  # Rename columns for correct format

    # Merge with the original dataframe to get 'Sport' and 'region'
    x = top_athletes.head(15).merge(df, left_on='index', right_on='Name', how='left')[['index', 'Name_x', 'Sport', 'region']].drop_duplicates('index')
    x.rename(columns={'index': 'Name', 'Name_x': 'Medals'}, inplace=True)
    return x


def year_wise_medal_tally_line(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                         ascending=False).reset_index()

    return final_df


def country_event_heatmap(df, country):
    # Remove rows where 'Medal' is NaN
    temp_df = df.dropna(subset=['Medal'])

    # Correctly dropping duplicates
    temp_df = temp_df.drop_duplicates(['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])

    # Filtering based on country
    new_df = temp_df[temp_df['region'] == country]

    # Counting occurrences instead of summing (ensuring 'Event' is used correctly)
    final_df = new_df.groupby(['Sport', 'Year'])['Event'].count().reset_index()

    # Creating pivot table
    pt = final_df.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='sum').fillna(0).astype(int)

    return pt


def most_successful_country_wise(df, country):
    temp_df = df.dropna(subset=['Medal'])  # Remove rows where Medal is NaN

    temp_df = temp_df[temp_df['region'] == country]  # Subset based on country
    top_athletes = temp_df['Name'].value_counts().reset_index()
    top_athletes.columns = ['Name', 'Medals']  # Rename columns

    # Merge with the original dataframe to get 'Sport' and 'region'
    x = top_athletes.head(10).merge(df, on='Name', how='left')[['Name', 'Medals', 'Sport']].drop_duplicates(
        'Name')  # Merge on 'Name' and extract relevant columns

    return x


def weight_v_height(df, sport):
    athelete_df = df.drop_duplicates(subset=['Name', 'region'])
    athelete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athelete_df[athelete_df['Sport'] == sport]
        return temp_df
    else:
        return athelete_df

def men_vs_women(df):
    athelete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athelete_df[athelete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athelete_df[athelete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)
    final.fillna(0, inplace=True)

    return final


































