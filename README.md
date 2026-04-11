# Time-series-weather-forecast-VN-province
## feature specification

### Original Base Features
These features come directly from the raw hourly data:

temperature: The air temperature at that specific hour.
humidity: The relative humidity percentage at that specific hour.
wind: The wind speed at that specific hour.
pressure: The atmospheric pressure at that specific hour.

### Time-Based Features

These are engineered from the original datetime column to help the model capture seasonality and daily patterns:

hour: The hour of the day (0 to 23). Captures daily cyclical patterns (e.g., warmer in the afternoon).
dayofweek: The day of the week (0 = Monday, 6 = Sunday). Might capture weekly cyclical patterns, although less relevant for weather than human behavior.
month: The month of the year (1 to 12). Crucial for capturing macro-level seasonal changes (e.g., dry season vs. rainy season).


### Lag and Rolling Window Features (Rainfall History)
These are engineered to give the model context about recent rainfall events, calculated independently for each city:

rain_lag1: The amount of rain that fell exactly 1 hour ago.
rain_lag3: The amount of rain that fell exactly 3 hours ago.
rain_lag24: The amount of rain that fell exactly 24 hours ago (helps capture daily recurring patterns).
rain_roll3: The average rainfall over the previous 3 hours (excluding the current hour). This provides a smoothed, short-term trend of recent rain.
rain_roll24: The average rainfall over the previous 24 hours (excluding the current hour). This provides a smoothed, longer-term trend of yesterday's rain.

### Categorical Features

city_* (e.g., city_hanoi, city_hcm): One-hot encoded (dummy) variables representing the city. The script drops the first city alphabetically to avoid the "dummy variable trap" (multicollinearity). For example, if there are 3 cities, there will be 2 city_ columns.
The Target Variable (What the model predicts)
target: The rainfall (rain column) shifted backward by 1 hour. This means the model is being trained to predict the amount of rain that will fall in the next hour for that specific city.