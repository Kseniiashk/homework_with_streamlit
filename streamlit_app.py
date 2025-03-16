import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def calculate_moving_average(data, window=30):
    data['moving_avg'] = data['temperature'].rolling(window=window).mean()
    data['moving_std'] = data['temperature'].rolling(window=window).std()
    return data

def detect_anomalies(data):
    data['anomaly'] = np.where(
        (data['temperature'] > data['moving_avg'] + 2 * data['moving_std']) | 
        (data['temperature'] < data['moving_avg'] - 2 * data['moving_std']), 1, 0)
    return data

def get_current_temperature(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        st.error("Ошибка при получении данных от OpenWeatherMap API.")
        return None

def plot_time_series(data, city):
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'], data['temperature'], label='Температура')
    ax.scatter(data[data['anomaly'] == 1]['timestamp'], data[data['anomaly'] == 1]['temperature'], color='red', label='Аномалии')
    ax.set_title(f'Температура в городе {city}')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Температура (°C)')
    ax.legend()
    st.pyplot(fig)

def plot_seasonal_profiles(data, city):
    seasonal_avg = data.groupby('season')['temperature'].mean()
    seasonal_std = data.groupby('season')['temperature'].std()
    
    fig, ax = plt.subplots()
    seasonal_avg.plot(kind='bar', yerr=seasonal_std, ax=ax, capsize=4)
    ax.set_title(f'Сезонные профили температуры в городе {city}')
    ax.set_xlabel('Сезон')
    ax.set_ylabel('Температура (°C)')
    st.pyplot(fig)

def get_city_coords(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [data['coord']['lat'], data['coord']['lon']]
    else:
        st.error(f"Не удалось получить координаты для города {city}.")
        return None
        
def show_map(city, current_temp, api_key):
    coords = get_city_coords(api_key, city)
    if coords:
        map = folium.Map(location=coords, zoom_start=10)
        folium.Marker(
            location=coords,
            popup=f"Текущая температура: {current_temp}°C",
            icon=folium.Icon(color='red')
        ).add_to(map)
        folium_static(map)
    else:
        st.warning(f"Не удалось найти координаты для города {city}.")

def main():
    st.title("Анализ температурных данных и мониторинг текущей температуры")
    
    uploaded_file = st.file_uploader("Загрузите файл с историческими данными о температуре", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        cities = data['city'].unique()
        selected_city = st.selectbox("Выберите город", cities)
        
        city_data = data[data['city'] == selected_city]
        
        city_data = calculate_moving_average(city_data)
        city_data = detect_anomalies(city_data)
        
        st.subheader("Описательная статистика")
        st.write(city_data.describe())
        
        st.subheader("Временной ряд температуры с аномалиями")
        plot_time_series(city_data, selected_city)
        
        st.subheader("Сезонные профили температуры")
        plot_seasonal_profiles(city_data, selected_city)
        
        st.subheader("Мониторинг текущей температуры")
        api_key = st.text_input("Введите ваш API ключ OpenWeatherMap", type="password")
        if api_key:
            current_temp = get_current_temperature(api_key, selected_city)
            if current_temp is not None:
                st.write(f"Текущая температура в городе {selected_city}: {current_temp}°C")
                
                st.subheader("Интерактивная карта")
                show_map(selected_city, current_temp, api_key)
                
                current_season = city_data[city_data['timestamp'].dt.month == datetime.now().month]['season'].mode()[0]
                season_data = city_data[city_data['season'] == current_season]
                avg_temp = season_data['temperature'].mean()
                std_temp = season_data['temperature'].std()
                
                if avg_temp - 2 * std_temp <= current_temp <= avg_temp + 2 * std_temp:
                    st.write("Текущая температура находится в пределах нормы.")
                else:
                    st.write("Текущая температура является аномальной.")
            else:
                st.error("Не удалось получить текущую температуру.")
        else:
            st.warning("Введите API ключ для получения текущей температуры.")

if __name__ == "__main__":
    main()
