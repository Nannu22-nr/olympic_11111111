import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
from helper import medal_tally

df = pd.read_csv("athlete_events.csv")
region_df = pd.read_csv("noc_regions.csv")


df = preprocessor.preprocess(df, region_df)

st.sidebar.title("Olympics Medal Analysis")
st.sidebar.image("olympic_img.jpg")
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Overall Analysis', 'Medal Tally', 'Country-wise Analysis','Athlete wise Analysis')
)

if user_menu == 'Medal Tally':
    st.sidebar.header('Medal Tally')
    years, country = helper.conuntry_year_list(df)

    selected_year = st.sidebar.selectbox('Select year', years)
    selected_country = st.sidebar.selectbox('Select country', country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title('Overall Tally')
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title('Medal Tally' + str(selected_year) + 'Olympics')
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + ' Overall Performance')
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + ' Performance in ' + str(selected_year) + ' Olympics')

    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]
    st.title('Overall Statistics')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('◯‍◯‍◯‍ Editions')
        st.title(editions)

    with col2:
        st.header('🏙️ Host Cities')
        st.title(cities)

    with col3:
        st.header('🏅 Sports')
        st.title(sports)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.header('🎯 Events')
        st.title(events)

    with col5:
        st.header('👥 Athletes')
        st.title(athletes)

    with col6:
        st.header('🌎 Nations')
        st.title(nations)



    nations_over_time= helper.participating_nations_over_time(df)
    fig = px.line(nations_over_time, x='Year', y='No of Countries', title="Number of Countries Participating Over Time")
    st.plotly_chart(fig)

    events_over_time = helper.participating_nations_over_time_1(df)
    fig = px.line(events_over_time, x='Year', y='Event', title="Number of Countries Participating Over Time")
    st.plotly_chart(fig)

    atheletes_over_time = helper.participating_nations_over_time_2(df)
    fig = px.line(atheletes_over_time, x='Year', y='Name', title="Number of Atheletes Participating Over Time")
    st.plotly_chart(fig)

    st.title('Number of Events Over Time(Every sports)')
    fig, ax = plt.subplots(figsize=(35, 25))

    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True)
    st.pyplot(fig)


    st.title('Most Successful Athletes')
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)


if user_menu == 'Country-wise Analysis':

    country_list = df['region'].unique().tolist()

    # Ensure all values are strings before sorting
    country_list = [str(country) for country in country_list]
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select Country', country_list)
    country_df = helper.year_wise_medal_tally_line(df, selected_country)

    st.title(selected_country + ' Medal number all time')
    fig = px.bar(country_df, x='region', y=['Gold', 'Silver', 'Bronze'], barmode='group')
    st.plotly_chart(fig)

    st.title(selected_country + ' Medals Tally over the years')
    pt = helper.country_event_heatmap(df, selected_country)
    fig, ax = plt.subplots(figsize=(35, 25))
    ax = sns.heatmap(pt, annot=True)
    st.pyplot(fig)

    st.title('top 10 alheletes ' + selected_country)
    top10_df = helper.most_successful_country_wise(df, selected_country)
    st.table(top10_df)


if user_menu == 'Athlete wise Analysis':
    athelete_df = df.drop_duplicates(subset=['Name', 'region'])
    # Clean the data to remove NaN and infinite values
    x1 = athelete_df['Age'].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    x2 = athelete_df[athelete_df['Medal'] == 'Gold']['Age'].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    x3 = athelete_df[athelete_df['Medal'] == 'Silver']['Age'].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    x4 = athelete_df[athelete_df['Medal'] == 'Bronze']['Age'].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    st.title('Athlete Age Distribution')
    # Create the distribution plot
    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],

                               show_hist=False, show_rug=False)
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']

    for sport in famous_sports:
        temp_df = athelete_df[athelete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_rug= False, show_hist= False)
    fig.update_layout(autosize = False, width = 1000, height = 600)
    st.title('Age distribution for GOLD medalist')
    st.plotly_chart(fig)

    for sport in famous_sports:
        temp_df = athelete_df[athelete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Silver']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_rug= False, show_hist= False)
    fig.update_layout(autosize = False, width = 1000, height = 600)
    st.title('Age distribution for SILVER medalist')
    st.plotly_chart(fig)

    import plotly.figure_factory as ff
    import streamlit as st

    # Assuming you have athlete_df and famous_sports defined elsewhere

    x = []
    name = []

    for sport in famous_sports:
        temp_df = athelete_df[athelete_df['Sport'] == sport]
        bronze_ages = temp_df[temp_df['Medal'] == 'Bronze']['Age'].dropna()

        if not bronze_ages.empty:  # Check if the Series is not empty
            x.append(bronze_ages)
            name.append(sport)
        else:
            print(f"No Bronze medalists found for {sport}, skipping.")  # Optional: Print a message

    fig = ff.create_distplot(x, name, show_rug=False, show_hist=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title('Age distribution for BRONZE medalist')
    st.plotly_chart(fig)

    st.title('Weight_VS_Height and Sex')
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')  # Corrected index to 0

    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)

    if not temp_df.empty:  # Check if DataFrame is not empty
        fig = px.scatter(
            temp_df,
            x='Weight',
            y='Height',
            color='Medal',
            symbol='Sex'
        )
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for {selected_sport}")


    st.title('Men Vs Women participation in different years')
    final = helper.men_vs_women(df)
    fig = px.line(final, x='Year', y=['Male', 'Female'])
    st.plotly_chart(fig)

