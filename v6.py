import pandas as pd
import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas as gpd
import plotly.express as px
from datetime import datetime


st.set_page_config(layout = 'wide')

@st.cache_data()
def get_data(path):
    data = pd.read_csv(path)
    
    return data


@st.cache_data()

def get_geofile(url):
    geofile = gpd.read_file( url)

    return geofile


def set_feature(data):
    data['price_m2'] = data['price']/ data['sqft_lot']
    data['price_m2'] = data['price_m2'].round(2)

    return data

def overview_data(data):
    
    f_attributes = st.sidebar.multiselect('Enter columns',data.columns)
    f_zipcode =  st.sidebar.multiselect('Enter ZipCode',data['zipcode'].unique())


    st.title('Data Overview')

    if (f_zipcode != [])& (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != [])& (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode)]

    elif (f_zipcode == [])& (f_attributes != []):
        data = data.loc[:,f_attributes]
    else:
        data = data.copy()


    st.dataframe(data)

    #Divisão de tabelas para o layout
    c1, c2 = st.columns((1.5,1))


    #===== Average Metrics =====
    df1 = data[['id','zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living','zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2','zipcode']].groupby('zipcode').mean().reset_index()

    #===== Merge =====
    m1 = pd.merge(df1, df2, on= 'zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode',how='inner')
    df = pd.merge(m2, df4, on='zipcode',how='inner')


    df.columns = ['ZIPCODE','TOTAL HOUSES','PRICE','SQRT LIVING','PRICE/M2']


    st.write(f_attributes)
    st.write(f_zipcode)


    c1.header('Average Values')
    c1.dataframe(df, height=600)

    #Statistic Descriptive 
    num_attributes = data.select_dtypes(include = ['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis = 1).reset_index()
    df1.columns = ['Attributes','Max','Min','Mean','Median','Std']

    c2.header('Statistic Model')
    c2.dataframe(df1, height=600, width=800)

    return None


def portfolio_density(data, geofile):

    st.title('Region Overview')

    c1,c2 = st.columns((1,1))
    c1.header('Portifolio Density')
    df = data.sample(30)

    #===== Base Map - Folium =====
    density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean()],
            default_zoom_start = 15)


    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                    popup= 'Price R${0} on:{1}. Features:{2} sqft, {3} bedrooms, {4}bathrooms, year built: {5}'.format( row['price'],
                                                                                                                        row['date'],
                                                                                                                        row['sqft_living'],
                                                                                                                        row['bedrooms'],
                                                                                                                        row['bathrooms'],
                                                                                                                        row['yr_built'])).add_to(marker_cluster)



    with c1: 
        folium_static(density_map)
        return None

    # ===== Region Price Map =====

    c2.header( 'Price Density')
    df = data[['price','zipcode']].groupby('zipcode').mean().reset_index()

    df.columns = ['ZIP','PRICE']

    df = df.sample(10)
    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map= folium.Map( location=[data['lat'].mean(), data['long'].mean()],
            default_zoom_start = 15)


    region_price_map.choropleth( data = df,
                                geo_data = geofile,
                                columns = ['ZIP','PRICE'], 
                                key_on = 'feature.properties.ZIP',
                                fill_color = 'YlOrRd',
                                fill_opacity = 0.7, 
                                line_opacity = 0.2, 
                                legend_name = 'AVG PRICE')

    with c2: 
        folium_static( region_price_map)

        
def commercial_distribution(data):
    # ===== Distribuição dos Imoveis por categorias comerciais ====

    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')



    # ===== Filters =====

    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built',min_year_built,
                                                max_year_built,
                                                max_year_built)



    st.header('Average Pricer per Year Built')
    # ===== Average Price per Year =====

    #Data Selection 
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built','price']].groupby('yr_built').mean().reset_index()

    #Plot
    fig = px.line(df, x = 'yr_built', y= 'price')
    st.plotly_chart( fig, use_container_width= True )

    #===== Average Price per Day
    st.header('Average Price per Day')
    st.sidebar.subheader(' Select Max Date')

    #Filter
    data['date'] = pd.to_datetime(data['date']).dt.date
    min_date = data['date'].min()
    max_date = data['date'].max()
    f_date = st.sidebar.slider('Date', min_date, max_date, max_date)

    #data filtering
    df = data.loc[data['date']< f_date]
    df = df[['date','price']].groupby('date').mean().reset_index()

    #plot
    fig = px.line(df, x = 'date', y= 'price')
    st.plotly_chart( fig, use_container_width= True )

    # ===== Histogram =====

    st.header("Price Distribution")
    st.sidebar.subheader('Select Max Price')

    #Filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # Data Filtering
    f_price = st.sidebar.slider('Price',min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    #data_plot 
    fig = px.histogram( df, x='price', nbins= 50)
    st.plotly_chart(fig, use_container_width= True)

    return None


def attributes_distribution(data):
    # ====== Distribuição dos Imoveis por categorias físicas =====

    st.sidebar.title('Attributes Options')

    st.title('House Attributes')

    #Filters
    f_bedrooms = st.sidebar.selectbox('Max Number of Bedrooms',
                        sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max Number of Bathrooms',
                        sorted(set(data['bathrooms'].unique())))

    c1,c2 = st.columns(2)


    #House per bedrooms
    c1.header('Houses Per Bedrooms')
    df = data[data['bedrooms']<f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins = 19)
    c1.plotly_chart(fig, use_container_width = True )

    #House per bathrooms
    c2.header('Houses per Bathrooms')
    df = data[data['bathrooms']< f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins = 19)
    c2.plotly_chart(fig, use_container_width = True )


    #filter 
    f_floors = st.sidebar.selectbox( 'Max Number of Floor', 
                                sorted(set(data['floors'].unique())))

    f_waterview = st.sidebar.checkbox('Only Houses with Water View')

    c1, c2 = st.columns(2)

    #House per floors
    c1.header('Houses Per Floor')
    df = data[data['floors']< f_floors]
    fig = px.histogram(data, x='floors', nbins = 10)
    c1.plotly_chart(fig, use_container_width = True )

    #House per water views
    if f_waterview:
        df = data[data['waterfront']== 1]
    else: 
        df = data.copy()

    fig = px.histogram(data, x='waterfront', nbins = 10)
    c2.plotly_chart(fig, use_container_width = True )

if __name__ == '__main__':
    #ETL 
    #data extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path) 
    geofile = get_geofile(url)

#transformation
data = set_feature(data)

overview_data(data)

portfolio_density(data, geofile)

commercial_distribution(data)

attributes_distribution(data)









