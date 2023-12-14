#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly.subplots import make_subplots
from xgboost import XGBRegressor, DMatrix, train as xgb_train
import plotly.offline as pyo
import datetime
import matplotlib.dates as mdates
#%%
# Data Acquisition
# load data
file_path = 'Weather_forecast_data.csv'
data = pd.read_csv(file_path,delimiter=';')
#%%
data.head()
data.tail()

#%% 
# Data cleaning
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], utc=True)
# Sorting the DataFrame by the 'Date' column in ascending order
data = data.sort_values(by='Date', ascending=True)
#%%
# Selecting specified columns
selected_columns= ['Date', 'Vitesse du vent moyen 10 mn', 'Température (°C)', 'Humidité']
weather_data = data[selected_columns]

#%%
# Converting 'Date' to datetime format and setting it as the index
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data.set_index('Date', inplace=True)
# Displaying the first few rows of the modified DataFrame
print(weather_data.head())

#%%
# Checking for missing values
missing_values = weather_data.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)

#%%
# Data visualization
# Descriptive Statistics
descriptive_stats = weather_data.describe()
print(descriptive_stats)
print("Median:\n", weather_data.median())
print("Mode:\n", weather_data.mode().iloc[0])

# Correlation Analysis with Significance
correlation_matrix = weather_data.corr()
p_values = weather_data.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*correlation_matrix.shape)
significant_correlation = p_values < 0.05
print(correlation_matrix)
print("Significant Correlations:\n", significant_correlation)

# Visualization
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Temperature Trend
axs[0].plot(weather_data.index, weather_data['Température (°C)'], color='red')
axs[0].set_title('Temperature Trend')
axs[0].set_ylabel('Temperature (°C)')

# Humidity Trend
axs[1].plot(weather_data.index, weather_data['Humidité'], color='blue')
axs[1].set_title('Humidity Trend')
axs[1].set_ylabel('Humidity (%)')

# Wind Speed Trend
axs[2].plot(weather_data.index, weather_data['Vitesse du vent moyen 10 mn'], color='green')
axs[2].set_title('Wind Speed Trend')
axs[2].set_ylabel('Wind Speed (km/h)')

plt.tight_layout()
plt.savefig(r'weather_trends.svg')
plt.show()

# Heatmap of Correlation Matrix with Significance
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, mask=~significant_correlation, cmap='coolwarm')
plt.title('Correlation between Weather Parameters with Significance')
plt.show()

# %%
# interactive visualization 
# Resetting the index so 'Date' becomes a column (needed for Plotly)
weather_data_reset = weather_data.reset_index()

# Define the start and end date for the initial view
start_date = datetime.datetime(2023, 10, 1)
end_date = datetime.datetime(2023, 11, 1)

# Creating traces for Temperature, Humidity, and Wind Speed
trace1 = go.Scatter(
    x=weather_data_reset['Date'],
    y=weather_data_reset['Température (°C)'],
    mode='lines',
    name='Temperature',
    line=dict(color='red'),
    hoverinfo='x+y+name',
    hovertemplate='%{y} °C on %{x}<extra></extra>'
)

trace2 = go.Scatter(
    x=weather_data_reset['Date'],
    y=weather_data_reset['Humidité'],
    mode='lines',
    name='Humidity',
    line=dict(color='blue'),
    hoverinfo='x+y+name',
    hovertemplate='%{y}% on %{x}<extra></extra>'
)

trace3 = go.Scatter(
    x=weather_data_reset['Date'],
    y=weather_data_reset['Vitesse du vent moyen 10 mn'],
    mode='lines',
    name='Wind Speed',
    line=dict(color='green'),
    hoverinfo='x+y+name',
    hovertemplate='%{y} km/h on %{x}<extra></extra>'
)

# Updating layout with responsive and aesthetic enhancements
layout = go.Layout(
    title='Weather Trends in Hérault',
    title_x=0.5,
    xaxis=dict(
        title='Date',
        range=[start_date, end_date],
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(step='all', label='All')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    ),
    yaxis=dict(title='Measurements'),
    legend=dict(
        x=1.1,
        y=1,
        xanchor='left',
        yanchor='top',
        orientation="v"
    ),
    hovermode='closest',
    margin=dict(r=150),
    autosize=True,
    font=dict(size=12),
    paper_bgcolor="LightSteelBlue"
)

# Creating figure
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

# Displaying figure
fig.show()
# Save the figure as an HTML file
html_file_path = 'weather_trends.html'
fig.write_html(html_file_path, full_html=False, include_plotlyjs='cdn')



#%%
# Predictive Analysis
# Function to create time series features from datetime index
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Load data
file_path = 'Weather_forecast_data.csv'
data = pd.read_csv(file_path, delimiter=';')

# Data Cleaning
# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], utc=True)
data.set_index('Date', inplace=True)

# Sorting the DataFrame by the 'Date' column in ascending order
data = data.sort_values(by='Date', ascending=True)

# Select specified columns (assuming these columns exist in your data)
selected_columns = ['Vitesse du vent moyen 10 mn', 'Température (°C)', 'Humidité']
weather_data = data[selected_columns]
dates = weather_data.index
#%%
# Add time series features to the DataFrame
weather_data = create_features(weather_data)
weather_data.head()
#%%
# Prepare the dataset
X = weather_data.drop('Température (°C)', axis=1)
y = weather_data['Température (°C)']
#%% 
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%% 
# Split data into training and testing sets along with dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_scaled, y, dates, test_size=0.2, random_state=42)

# Train the XGBoost model (assuming XGBRegressor is used here)
xgb_model = XGBRegressor(n_estimators=1000, max_depth=3, eta=0.1, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# Predict using the model
y_pred = xgb_model.predict(X_test)

# %%
# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
# %%
# Prepare for future prediction
future_dates = pd.date_range(start='2023-11-30', end='2023-12-30', freq='3H')
future_data = pd.DataFrame(index=future_dates)

# Add the same time-related features as in the original dataset

future_data['Vitesse du vent moyen 10 mn'] = np.random.normal(10, 2, len(future_data))
future_data['Humidité'] = np.random.randint(30, 80, len(future_data))
future_data['hour'] = future_data.index.hour
future_data['dayofweek'] = future_data.index.dayofweek
future_data['quarter'] = future_data.index.quarter
future_data['month'] = future_data.index.month
future_data['year'] = future_data.index.year
future_data['dayofyear'] = future_data.index.dayofyear
future_data['dayofmonth'] = future_data.index.day
future_data['weekofyear'] = future_data.index.isocalendar().week

#%%
# Scale the future data
future_features_scaled = scaler.transform(future_data)
#%%
# Predict future temperatures
future_temps = xgb_model.predict(future_features_scaled)
# %%
# Save and visualize the future predictions
future_temps_df = pd.DataFrame({'Date': future_dates, 'Predicted_Temperature_C': future_temps})
csv_file_path = 'predicted_temperatures.csv'
future_temps_df.to_csv(csv_file_path, index=False)

#%%
# Visualization code for future predictions
# Convert 'Date' column back to datetime for plotting
future_temps_df['Date'] = pd.to_datetime(future_temps_df['Date'])

# Create a Plotly figure
fig = go.Figure()

# Add trace for predicted temperatures
fig.add_trace(
    go.Scatter(
        x=future_temps_df['Date'],
        y=future_temps_df['Predicted_Temperature_C'],
        mode='lines+markers',
        name='Predicted Temperature',
        line=dict(color='teal'),  # Changed to teal for a cool look
        marker=dict(size=4)
    )
)

# Update layout for aesthetics with grid lines
fig.update_layout(
    title='Future Predicted Temperature from 30 Nov to 30 Dec 2023 (Every 3 Hours)',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
    legend_title='Temperature Type',
    paper_bgcolor='LightSteelBlue', # Background color
    plot_bgcolor='white', # Plotting area background color
    hovermode='x unified', # Unified hover mode
    margin=dict(l=20, r=20, t=40, b=20), # Adjust margins
    font=dict(size=12), # Font size
    xaxis=dict(showgrid=True, gridcolor='lightgray'), # X-axis grid
    yaxis=dict(showgrid=True, gridcolor='lightgray'), # Y-axis grid
)

# Save the figure as an HTML file
html_file_path = 'predicted_future_temps.html'
fig.write_html(html_file_path, full_html=False, include_plotlyjs='cdn')

# Optional: Display the figure in your Python environment
fig.show()

# Output the path to the saved HTML file
print(f"Interactive graph saved to: {html_file_path}")
# Output the path to the saved CSV file
print(f"Predicted temperatures saved to: {csv_file_path}")

#%%
# Sort the test data by dates to avoid crossover lines in the plot
sorted_indices = np.argsort(dates_test)
sorted_dates_test = dates_test[sorted_indices]
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

# Interactive Visualization Actual vs Predicted Temperature : 

# Create a Plotly figure
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add trace for actual temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_test, name='Actual', mode='markers+lines', marker=dict(color='blue'), line=dict(width=2)),
    secondary_y=False,
)

# Add trace for predicted temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_pred, name='Predicted', mode='markers+lines', marker=dict(color='red'), line=dict(width=2)),
    secondary_y=False,
)

# Set figure layout
fig.update_layout(
    paper_bgcolor="LightSteelBlue",
    title_text='Actual vs Predicted Temperature',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
    legend_title='Legend',
    hovermode='x unified'
)

# Update y-axes titles
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)

# Show the figure
fig.show()
#%%


# Create a Plotly figure
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Define the start and end date for the initial view
start_date = datetime.datetime(2023, 10, 1)
end_date = datetime.datetime(2023, 11, 1)

# Update layout and axes with grid lines
fig.update_layout(
    title_text='Actual vs Predicted Temperature Over Time',
    xaxis=dict(
        title='Date',
        range=[start_date, end_date],
        gridcolor='lightgray',
        gridwidth=1,
        showgrid=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All Time")
            ]),
            bgcolor="lightblue",
            font=dict(size=12),
            x=0.1,
            y=1.2
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(
        title='Temperature (°C)',
        gridcolor='lightgray',
        gridwidth=1,
        showgrid=True
    ),
    plot_bgcolor='white',
    paper_bgcolor="LightSteelBlue",
    font=dict(family="Arial, sans-serif", size=14, color="darkslategray"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        x=1.1,
        y=1,
        xanchor='left',
        yanchor='top',
        orientation="v"
    ),
    hovermode='x unified'
)

# Add traces for actual and predicted temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_test, name='Actual Temperature', mode='markers+lines', marker=dict(color='blue', size=5), line=dict(width=3)),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_pred, name='Predicted Temperature', mode='markers+lines', marker=dict(color='red', size=5), line=dict(width=3)),
    secondary_y=False,
)



# Show the figure
fig.show()

# Assuming 'fig' is your Plotly figure
graph_html = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
# Specify the directory and file name where you want to save the HTML file
html_file_path = 'plot.html'
# Save the figure
with open(html_file_path, "w") as file:
    file.write(graph_html)
#%%

#%%
# Sort the test data by dates to avoid crossover lines in the plot
sorted_indices = np.argsort(dates_test)
sorted_dates_test = dates_test[sorted_indices]
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]
# Visualizing Actual vs Predicted Temperature with improved clarity
plt.figure(figsize=(15, 7))
plt.plot(sorted_dates_test, sorted_y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=5, alpha=0.7)
plt.plot(sorted_dates_test, sorted_y_pred, label='Predicted', color='red', marker='x', linestyle='-', markersize=5, alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()
#%%

#%%
# Creating a Plotly figure
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Adding trace for actual temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_test, name='Actual', mode='markers+lines', marker=dict(color='blue'), line=dict(width=2)),
    secondary_y=False,
)

# Adding trace for initial predicted temperatures
fig.add_trace(
    go.Scatter(x=sorted_dates_test, y=sorted_y_pred, name='Initial Predicted', mode='markers+lines', marker=dict(color='red'), line=dict(width=2)),
    secondary_y=False,
)

# Adding trace for future predicted temperatures
fig.add_trace(
    go.Scatter(x=future_temps_df['Date'], y=future_temps_df['Predicted_Temperature_C'], name='Future Predicted', mode='markers+lines', marker=dict(color='green'), line=dict(width=2)),
    secondary_y=False,
)

# Setting figure layout
fig.update_layout(
    title='Actual vs Predicted Temperature',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
    legend_title='Legend',
    hovermode='x unified'
)

# Updating y-axis titles
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)

# Adding range slider and selector
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Showing the figure
fig.show()

#%%

# Convert sorted_dates_test and future_temps_df['Date'] to datetime if not already
sorted_dates_test = pd.to_datetime(sorted_dates_test)
future_temps_df['Date'] = pd.to_datetime(future_temps_df['Date'])
#%%
# Define date ranges for the plots
dec_dates = (sorted_dates_test >= '2023-11-23') & (sorted_dates_test <= '2023-11-29')
jan_dates = (sorted_dates_test >= '2023-01-01') & (sorted_dates_test <= '2023-01-31')
jul_dates = (sorted_dates_test >= '2023-07-01') & (sorted_dates_test <= '2023-07-07')

# Create a subplot for each date range
fig = make_subplots(rows=3, cols=1, subplot_titles=('Last Week of November', 'First Month of January', 'First Week of July'))

# Function to add traces to a subplot with different colors and legend names
def add_traces(row, dates_range, actual_color, predicted_color, actual_legend, predicted_legend):
    # Actual Temperature
    fig.add_trace(
        go.Scatter(x=sorted_dates_test[dates_range], y=sorted_y_test[dates_range], name=actual_legend, mode='lines', line=dict(color=actual_color)),
        row=row, col=1
    )
    # Predicted Temperature
    fig.add_trace(
        go.Scatter(x=sorted_dates_test[dates_range], y=sorted_y_pred[dates_range], name=predicted_legend, mode='lines', line=dict(color=predicted_color)),
        row=row, col=1
    )

# Adding traces for each date range with unique colors and legend names
add_traces(1, dec_dates, 'navy', 'maroon', 'Actual - Nov', 'Predicted - Nov')
add_traces(2, jan_dates, 'darkgreen', 'darkorange', 'Actual - Jan', 'Predicted - Jan')
add_traces(3, jul_dates, 'purple', 'gold', 'Actual - Jul', 'Predicted - Jul')

# Update layout
fig.update_layout(height=900, width=700, title_text="Weather Data Analysis", showlegend=True)

# Show the figure
fig.show()

#%%
# Load the pollution data into a DataFrame 
# Change with your actual path
# unable to unset the absolute path
weather_file_path = 'C:/Users/SCD-UM//OneDrive/Bureau/My_Pollution_project/codes/Mesure_Annuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(weather_file_path, delimiter=',')
df.head()
#%%

#%%
# Pivot the DataFrame
pivot_df = df.pivot_table(index='nom_com', columns='nom_poll', values='valeur', aggfunc='mean').reset_index()

# Plotting the comparative bar chart
fig_city_pollutants = px.bar(
    pivot_df,
    x='nom_com',
    y=pivot_df.columns[1:],  # Assuming first column is 'nom_com'
    title='Comparaison des niveaux de différents polluants par ville',
    labels={'value': 'Niveau moyen du polluant', 'variable': 'Polluant', 'nom_com': 'Ville'},
    color_discrete_sequence=px.colors.qualitative.Bold  # Use a vibrant color sequence
)

# Update layout for background color and text readability
fig_city_pollutants.update_layout(
    barmode='group',
    xaxis_tickangle=-45,
    plot_bgcolor='white',  # Set the plotting area background
    paper_bgcolor='LightSteelBlue',  # Set the overall figure background
    font=dict(color='black'),  # Set text color for better readability
    title=dict(font=dict(size=16, color='black')),  # Title styling
    legend=dict(font=dict(color='black'))  # Legend text color
)


fig_city_pollutants.update_layout(barmode='group', xaxis_tickangle=-45)
fig_city_pollutants.write_html('city_pollutants_comparison.html', full_html=False, include_plotlyjs='cdn')
fig_city_pollutants.show()

#%%
# Fill missing values in 'valeur' with its mean
df['valeur'] = df['valeur'].fillna(df['valeur'].mean())

# For 'nom_dept' and 'nom_com', decide how to handle missing values. 
# If they're categorical, you might not want to fill them with mean.
# You could fill them with a placeholder like 'Unknown', or decide based on your context.

df['nom_dept'] = df['nom_dept'].fillna('Unknown')
df['nom_com'] = df['nom_com'].fillna('Unknown')

# Convert date columns to datetime
df['date_debut'] = pd.to_datetime(df['date_debut'])
df['date_fin'] = pd.to_datetime(df['date_fin'])
#%%
# Calculating AQI (This is a simplified example. Actual AQI calculation might be more complex)
# For simplicity, I'm considering higher 'valeur' as worse air quality
def calculate_aqi(value):
    if value <= 30:
        return "Good"
    elif value <= 60:
        return "Moderate"
    elif value <= 90:
        return "Unhealthy for Sensitive Groups"
    elif value <= 120:
        return "Unhealthy"
    elif value <= 150:
        return "Very Unhealthy"
    else:
        return "Hazardous"

df['AQI'] = df['valeur'].apply(calculate_aqi)

# Grouping by department and city to get the average value
aqi_dept = df.groupby('nom_dept')['valeur'].mean().reset_index()
aqi_city = df.groupby('nom_com')['valeur'].mean().reset_index()

# Showing the first few rows of the AQI data for departments and cities
aqi_dept.head(), aqi_city.head()
#%%
# Custom color scale based on AQI levels
custom_color_scale = [
    (0.00, "green"),  # Good
    (0.30, "yellow"), # Moderate
    (0.60, "orange"), # Unhealthy for Sensitive Groups
    (1.00, "red")     # Unhealthy and above
]

# AQI Bar Chart by Department
fig_dept = px.bar(aqi_dept, x='nom_dept', y='valeur', 
                  title='Indice de Qualité de l\'Air par Département',
                  labels={'valeur': 'AQI Moyen', 'nom_dept': 'Département'},
                  color='valeur',
                  color_continuous_scale=custom_color_scale,
                  hover_data={'nom_dept': True, 'valeur': True})

# Enhancements
fig_dept.update_layout(transition_duration=500)  # Animation
fig_dept.update_traces(hovertemplate="Département: %{x}<br>AQI Moyen: %{y}")
fig_dept.update_layout(legend_title_text='AQI', 
                       xaxis_title='Département', 
                       yaxis_title='Indice de Qualité de l\'Air',
                       font=dict(family="Arial, sans-serif", size=12, color="RebeccaPurple"))

# Adding more detailed tooltips
fig_dept.update_traces(hovertemplate="<b>%{x}</b><br>AQI Moyen: %{y}<br>Plus d'infos...")

# Adding a trend line
fig_dept.add_traces(go.Scatter(x=aqi_dept['nom_dept'], y=np.polyval(np.polyfit(range(len(aqi_dept)), aqi_dept['valeur'], 1), range(len(aqi_dept))), mode='lines', name='Tendance'))

# Improving layout for mobile responsiveness
fig_dept.update_layout(autosize=True)
# Set LightSteelBlue as the background color
fig_dept.update_layout(
    paper_bgcolor='LightSteelBlue'
)

# Save to HTML
fig_dept.write_html('aqi_dept_chart.html', full_html=False, include_plotlyjs='cdn')
fig_dept.show()
#%%
# Define a more detailed color scale for AQI levels
aqi_color_scale = [
    (0.0, "green"),  # Good
    (0.2, "yellowgreen"),  # Moderate
    (0.4, "yellow"),  # Unhealthy for Sensitive Groups
    (0.6, "orange"),  # Unhealthy
    (0.8, "red"),  # Very Unhealthy
    (1.0, "purple")  # Hazardous
]


# AQI Bar Chart by City
fig_city = px.bar(aqi_city, x='nom_com', y='valeur', 
                  title='Indice de Qualité de l\'Air par Ville',
                  labels={'valeur': 'AQI Moyen', 'nom_com': 'Ville'},
                  color='valeur',
                  color_continuous_scale=custom_color_scale,
                  hover_data={'nom_com': True, 'valeur': True})

# Enhancements
fig_city.update_layout(transition_duration=500)  # Animation
fig_city.update_traces(hovertemplate="Ville: %{x}<br>AQI Moyen: %{y}")
fig_city.update_layout(
    legend=dict(title=dict(text='City'), itemsizing='constant'),
    xaxis_title='Ville', 
    yaxis_title='Indice de Qualité de l\'Air',
    font=dict(family="Arial, sans-serif", size=12, color="RebeccaPurple"))

# Set LightSteelBlue as the background color
fig_city.update_layout(
    paper_bgcolor='LightSteelBlue'
)

# Save to HTML
fig_city.write_html('aqi_city_chart.html', full_html=False, include_plotlyjs='cdn')
fig_city.show()
#%%
# Convert 'date_debut' to datetime and extract the month
df['date_debut'] = pd.to_datetime(df['date_debut'])
df['month'] = df['date_debut'].dt.month

# Define a function to map months to seasons
def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Apply the function to create a 'season' column
df['season'] = df['month'].apply(month_to_season)

# Group by season and calculate the average pollution value
seasonal_avg = df.groupby('season')['valeur'].mean().reset_index()

# Sort the seasons in order
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
seasonal_avg['season'] = pd.Categorical(seasonal_avg['season'], categories=season_order, ordered=True)
seasonal_avg = seasonal_avg.sort_values('season')

# Create a line graph or area chart
fig = px.line(seasonal_avg, x='season', y='valeur', title='Seasonal Variation of Pollution Levels',
              labels={'valeur': 'Average Pollution Value', 'season': 'Season'})

# Enhanced Styling
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=seasonal_avg['season'], 
    y=seasonal_avg['valeur'], 
    mode='lines+markers',
    name='Avg Pollution',
    line=dict(color='royalblue', width=4),
    marker=dict(color='lightblue', size=10)
))

# Adding title and labels
fig.update_layout(
    title='Seasonal Variation of Pollution Levels',
    xaxis_title='Season',
    yaxis_title='Average Pollution Value',
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=12, color='black')
)

# Adding Annotations for Better Context
fig.add_annotation(
    x=seasonal_avg['season'].iloc[seasonal_avg['valeur'].idxmax()],
    y=seasonal_avg['valeur'].max(),
    text='Peak Pollution',
    showarrow=True,
    arrowhead=1
)

# Make the figure responsive
fig.update_layout(autosize=True)

# Show the figure
fig.show()

# Save to HTML
fig.write_html('seasonal_variation_chart.html', full_html=False, include_plotlyjs='cdn')
#%%
# Load the pollution data into a DataFrame 
# Change with your actual path
# unable to unset the absolute path
weather_file_path = 'C:/Users/SCD-UM//OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/Mesure_Annuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(weather_file_path, delimiter=',')
df.head()

# Check if the date columns need to be converted to datetime
if not pd.api.types.is_datetime64_any_dtype(df['date_debut']):
    df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')

# Extract the year from the date
df['year'] = df['date_debut'].dt.year

# Group by year and calculate the average pollution value
yearly_avg = df.groupby('year')['valeur'].mean().reset_index()

# Create a scatter plot with lines for detailed points
scatter = go.Scatter(
    x=yearly_avg['year'],
    y=yearly_avg['valeur'],
    mode='markers+lines',
    marker=dict(color='blue', size=8),
    name='Average Pollution Level',
    hoverinfo='text',
    text=yearly_avg['year'].astype(str) + ': ' + yearly_avg['valeur'].round(2).astype(str)
)

# Add a trend line (3-year moving average)
trendline = go.Scatter(
    x=yearly_avg['year'],
    y=yearly_avg['valeur'].rolling(window=3).mean(),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Trend Line'
)

# Create figure and add traces
fig = go.Figure()
fig.add_trace(scatter)
fig.add_trace(trendline)


# Set LightSteelBlue as the background color
fig.update_layout(
    title='Historical Pollution Trend Analysis with Trend Line',
    xaxis=dict(title='Year', tickmode='linear'),
    yaxis=dict(title='Average Pollution Level'),
    hovermode='x unified',
    template='plotly_white',
    autosize=True,
    margin=dict(l=50, r=50, b=50, t=50),
    paper_bgcolor='LightSteelBlue'  # Set the background color
)

# Save the figure to HTML
fig.write_html('air_quality_trend_analysis.html', full_html=False, include_plotlyjs='cdn')

# Optional: Display the figure in your Python environment
fig.show()

#%%


#%%
# Describe the 'nom_dept' column
nom_dept_description = df['nom_dept'].describe()
nom_dept_description
#%%
# Display unique values in 'nom_dept' column
unique_values = df['nom_dept'].unique()
unique_values
#%%
# Filter the DataFrame for rows where 'nom_dept' is 'HERAULT'
df_HERAULT = df[df['nom_dept'] == 'HERAULT']

# Display the first few rows of the new DataFrame
df_HERAULT.head()
# %%
df_HERAULT.describe()
#%%
unique_values = data['communes (name)'].unique()
unique_values

# %%
# Save df_HERAULT as a CSV file
output_file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/df_HERAULT.csv'
df_HERAULT.to_csv(output_file_path, index=False)
# %%
# Load the weather data
weather_file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/Weather_forecast_data.csv'
weather_data = pd.read_csv(weather_file_path, delimiter=';')

# Ensure 'date_debut' in df_HERAULT and 'Date' in weather_data are datetime objects
df_HERAULT['date_debut'] = pd.to_datetime(df_HERAULT['date_debut'])
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Set 'date_debut' and 'Date' as the indices
df_HERAULT.set_index('date_debut', inplace=True)
weather_data.set_index('Date', inplace=True)

# Merge the datasets
combined_data = pd.merge(df_HERAULT, weather_data, left_index=True, right_index=True, how='inner')

# Check the combined data
print(combined_data.head())
#%%
# Convert 'Date' column to datetime and set as index for weather_data
weather_data['Date'] = pd.to_datetime(weather_data['Date'], utc=True)
weather_data.set_index('Date', inplace=True)

# Convert the index of weather_data to timezone-naive
weather_data.index = weather_data.index.tz_localize(None)

# Identify and exclude non-numeric columns, then resample
non_numeric_columns = weather_data.select_dtypes(include=['object']).columns
monthly_weather = weather_data.drop(columns=non_numeric_columns).resample('M').mean()

# Ensure df_HERAULT has the correct datetime index
df_HERAULT['date_debut'] = pd.to_datetime(df_HERAULT['date_debut'])
df_HERAULT.set_index('date_debut', inplace=True)
df_HERAULT.index = df_HERAULT.index.to_period('M').to_timestamp('M')

# Merge the datasets
combined_data = pd.merge(df_HERAULT, monthly_weather, left_index=True, right_index=True, how='inner')

#%%

# Ensure that the index is a DatetimeIndex
if not isinstance(df_HERAULT.index, pd.DatetimeIndex):
    df_HERAULT.index = pd.to_datetime(df_HERAULT.index)

# Convert the index to 'period[M]' and then back to timestamp at month end
df_HERAULT.index = df_HERAULT.index.to_period('M').to_timestamp('M')

# Now, perform the merge
combined_data = pd.merge(df_HERAULT, monthly_weather, left_index=True, right_index=True, how='inner')

# Check the combined data
combined_data.head()
# %%
# Check for missing values
missing_values = combined_data.isnull().sum()
missing_values
#%%
# Explore data statistics
combined_stats = combined_data.describe()
combined_stats
# %%
# Print column names of the combined_data DataFrame
print(combined_data.columns)
#%%
# Assuming df_HERAULT and weather_data are your original datasets

# Reset index of df_HERAULT if 'date_debut' is set as index
if isinstance(df_HERAULT.index, pd.DatetimeIndex):
    df_HERAULT = df_HERAULT.reset_index()

# Now merge df_HERAULT with weather_data
# Assuming the weather data is already aggregated to the same time frequency (e.g., monthly)
combined_data = pd.merge(df_HERAULT, weather_data, left_on='date_debut', right_index=True, how='inner')

# Check if 'date_debut' is now in the combined_data
print(combined_data.columns)

# Proceed with selecting your relevant columns
relevant_columns = ['date_debut', 'nom_poll', 'valeur', 'date_fin', 'Température', 'Humidité', 
                    'Vitesse du vent moyen 10 mn', 'nom_station', 'typologie', 'influence']
sub_dataframe = combined_data[relevant_columns]
#%%
# Display the first few rows of the sub-dataframe
sub_dataframe.head()

#%%

# %%
# Check for missing values
missing_values = sub_dataframe.isnull().sum()
print("Missing Values:\n", missing_values)

#%%
# Identify numeric columns
numeric_cols = sub_dataframe.select_dtypes(include=[np.number]).columns

# Fill missing values in numeric columns with the mean of their respective column
sub_dataframe[numeric_cols] = sub_dataframe[numeric_cols].fillna(sub_dataframe[numeric_cols].mean())

# Check for missing values again
missing_values_after = sub_dataframe.isnull().sum()
print("Missing Values after filling with mean:\n", missing_values_after)
# %%
# Basic descriptive statistics
descriptive_stats = sub_dataframe.describe()
print(descriptive_stats)
#%%
sub_dataframe.head()
#%%

# %%

# Descriptive Statistics
numeric_columns = ['valeur', 'Température', 'Humidité', 'Vitesse du vent moyen 10 mn']
descriptive_stats = sub_dataframe[numeric_columns].describe()
print(descriptive_stats)



# Correlation Analysis
correlation_matrix = sub_dataframe[numeric_columns].corr()
print(correlation_matrix)

# Heatmap of Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Weather Parameters and Pollution Levels')
plt.show()

# %%
# Visualization
sub_dataframe['date_debut'] = pd.to_datetime(sub_dataframe['date_debut'])
def create_dual_axis_plot(x, y1, y2, y1_label, y2_label, y1_color, y2_color, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting the first parameter
    ax1.plot(x, y1, color=y1_color, label=y1_label)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    
    # Format x-axis to show month and year
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.tick_params(axis='x', rotation=45)

    # Creating a second y-axis for pollution values
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=y2_color, label='Pollution Level')
    ax2.set_ylabel('Pollution Value', color=y2_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)

    # Adding legend for both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Set title
    plt.title(title)

    plt.show()

# Now, call the function for each trend
# Temperature vs Pollution Levels
create_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Température'], sub_dataframe['valeur'],
                      'Temperature', 'Pollution Value', 'red', 'purple', 'Temperature vs Pollution Levels')

# Humidity vs Pollution Levels
create_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Humidité'], sub_dataframe['valeur'],
                      'Humidity', 'Pollution Value', 'blue', 'purple', 'Humidity vs Pollution Levels')

# Wind Speed vs Pollution Levels
create_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Vitesse du vent moyen 10 mn'], sub_dataframe['valeur'],
                      'Wind Speed', 'Pollution Value', 'green', 'purple', 'Wind Speed vs Pollution Levels')
# %%
def create_interactive_dual_axis_plot(x, y1, y2, y1_label, y2_label, y1_color, y2_color, title):
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the first plot (e.g., Temperature)
    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1_label, mode='lines', line=dict(color=y1_color)),
        secondary_y=False,
    )

    # Add the second plot (e.g., Pollution Level)
    fig.add_trace(
        go.Scatter(x=x, y=y2, name='Pollution Level', mode='lines', line=dict(color=y2_color)),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text=title)

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text=y1_label, secondary_y=False)
    fig.update_yaxes(title_text=y2_label, secondary_y=True)

    # Customize layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )

    fig.show()

# Temperature vs Pollution Levels
create_interactive_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Température'], sub_dataframe['valeur'],
                                  'Temperature (°C)', 'Pollution Value', 'red', 'purple',
                                  'Interactive Plot: Temperature vs Pollution Levels')

# Humidity vs Pollution Levels
create_interactive_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Humidité'], sub_dataframe['valeur'],
                                  'Humidity (%)', 'Pollution Value', 'blue', 'purple',
                                  'Interactive Plot: Humidity vs Pollution Levels')

# Wind Speed vs Pollution Levels
create_interactive_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Vitesse du vent moyen 10 mn'], sub_dataframe['valeur'],
                                  'Wind Speed (km/h)', 'Pollution Value', 'green', 'purple',
                                  'Interactive Plot: Wind Speed vs Pollution Levels')

# %%
# Correlation study : 
# Perform a correlation analysis
correlation_matrix = sub_dataframe[['Température', 'Humidité', 'Vitesse du vent moyen 10 mn', 'valeur']].corr()

# Compute p-values for the correlations
p_values = sub_dataframe[['Température', 'Humidité', 'Vitesse du vent moyen 10 mn', 'valeur']].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*correlation_matrix.shape)

# Define a function to annotate the heatmap with the correlation coefficient and its significance
def heatmap_with_annotations(data, p_values, threshold=0.05):
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(data, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')
    for i, j in zip(*np.where(p_values < threshold)):
        heatmap.text(j+0.5, i+0.5, "*", ha='center', va='center', color='white')
    plt.title('Correlation between Weather Parameters and Pollution Levels')
    plt.show()

# Call the function
heatmap_with_annotations(correlation_matrix, p_values)

# %%
# # %%
# Plotting the data
# Example 1: Line plot for temperature over time.
plt.figure(figsize=(10, 5))
sns.lineplot(data=sub_dataframe, x='date_debut', y='Température', marker='o')
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Example 2: Bar plot for pollution values by type.
plt.figure(figsize=(10, 5))
sns.barplot(data=sub_dataframe, x='nom_poll', y='valeur', hue='influence', dodge=False)
plt.title('Pollution Values by Type')
plt.xlabel('Pollutant')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# Example 3: Scatter plot for humidity vs. temperature.
plt.figure(figsize=(10, 5))
sns.scatterplot(data=sub_dataframe, x='Température', y='Humidité', hue='nom_poll', style='nom_poll', s=100)
plt.title('Humidity vs. Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Interactive line plot for temperature over time.
fig = px.line(sub_dataframe, x='date_debut', y='Température', title='Temperature Over Time')
# %%
fig.show()
#%%

#%%
import pandas as pd
from ipyleaflet import Map, TileLayer, GeoJSON, Marker
import ipywidgets as widgets
from IPython.display import display
import numpy as np 
#%%
# Load CSV data into DataFrame
chemin_fichier_csv = 'C:/Users/SCD-UM/OneDrive/Bureau/My_Pollution_project/codes/Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(chemin_fichier_csv)

# Create 'geometry' column in GeoJSON format
df['geometry'] = df.apply(lambda row: {"type": "Point", "coordinates": [row['X'], row['Y']]}, axis=1)

# Threshold value
SEUIL_DE_VALEUR_ELEVEE = 23

# Replace infinite values with NaN and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Ensure 'X' (longitude) and 'Y' (latitude) are within valid range
df = df[(df['X'].between(-180, 180)) & (df['Y'].between(-90, 90))]

# Set the initial center of the map to your area of interest
# Example: Centering on Toulouse, Occitanie
center_latitude = 43.6045  # Replace with the latitude of your area
center_longitude = 1.4442  # Replace with the longitude of your area
initial_zoom_level = 9      # Adjust the zoom level to your preference

# Create the map with the specified center and zoom level
carte = Map(center=(center_latitude, center_longitude), zoom=initial_zoom_level)

# Add WMTS layer to the map
wmts_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS"
wmts_layer = TileLayer(url=wmts_url, name="WMTS Layer")
carte.add_layer(wmts_layer)

# Créer une GeoJSON FeatureCollection à partir des données de votre DataFrame
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Marqueurs pour les valeurs élevées
high_value_markers = []

for index, row in df.iterrows():
    feature = {
        "type": "Feature",
        "geometry": row['geometry'],
        "properties": {"nom_dept": row['nom_dept'], "valeur": row['valeur']}
    }
    geojson_data['features'].append(feature)

    # Ajouter un marqueur si la valeur de pollution est élevée
    if row['valeur'] > SEUIL_DE_VALEUR_ELEVEE:
        marker = Marker(location=(row['Y'], row['X']), draggable=False, title=f"Valeur: {row['valeur']}")
        high_value_markers.append(marker)

def pollution_color(valeur):
    """Function to determine color based on pollution level."""
    if valeur > 50:
        return "red"
    elif valeur > 30:
        return "orange"
    elif valeur > 10:
        return "yellow"
    else:
        return "green"

def create_geojson_layer(df, threshold):
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    for _, row in df.iterrows():
        color = pollution_color(row['valeur'])
        feature = {
            "type": "Feature",
            "geometry": row['geometry'],
            "properties": {"nom_dept": row['nom_dept'], "valeur": row['valeur'], "style": {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.8}}
        }
        geojson_data['features'].append(feature)
    return geojson_data
# Créer une couche GeoJSON pour les zones polluées
# Update the GeoJSON layer creation in your main code
geojson_layer = create_geojson_layer(df, SEUIL_DE_VALEUR_ELEVEE)
carte.add_layer(geojson_layer)

# Update the legend to reflect new categories
legend = widgets.VBox([
    widgets.HTML(value="<b>Légende</b>"),
    widgets.HTML(value='<div style="background:red; width:24px; height:24px; display:inline-block;"></div> Très Élevée (> 50)'),
    widgets.HTML(value='<div style="background:orange; width:24px; height:24px; display:inline-block;"></div> Élevée (> 30)'),
    widgets.HTML(value='<div style="background:yellow; width:24px; height:24px; display:inline-block;"></div> Modérée (> 10)'),
    widgets.HTML(value='<div style="background:green; width:24px; height:24px; display:inline-block;"></div> Faible (<= 10)')
])




# Ajouter les marqueurs à la carte

for marker in high_value_markers:
    carte.add_layer(marker)


# Afficher la carte avec la légende
display(widgets.HBox([carte, legend]))
#%%

# %%

import pandas as pd
from ipyleaflet import Map, Marker, Popup
import ipywidgets as widgets

# Load data
df = pd.read_csv('C:/Users/SCD-UM/OneDrive/Bureau/My_Pollution_project/codes/Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv')
df['geometry'] = df.apply(lambda row: {"type": "Point", "coordinates": [row['X'], row['Y']]}, axis=1)

# Define your threshold value
SEUIL_DE_VALEUR_ELEVEE = 23

# Create a map centered around Occitanie
carte = Map(center=(43.611015, 3.876733), zoom=9)

# Add markers to the map
for _, row in df.iterrows():
    if row['valeur'] > SEUIL_DE_VALEUR_ELEVEE:
        marker = Marker(location=(row['Y'], row['X']))
        popup = Popup(child=widgets.HTML(f"Valeur: {row['valeur']}"))
        marker.popup = popup
        carte.add_layer(marker)

# Display the map
display(carte)


# %%
import pandas as pd
from ipyleaflet import Map, TileLayer, GeoJSON, Marker, Heatmap
import ipywidgets as widgets
from IPython.display import display
import numpy as np 

# Load CSV data into DataFrame
chemin_fichier_csv = 'C:/Users/SCD-UM/OneDrive/Bureau/My_Pollution_project/codes/Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(chemin_fichier_csv)

# Create 'geometry' column in GeoJSON format
df['geometry'] = df.apply(lambda row: {"type": "Point", "coordinates": [row['X'], row['Y']]}, axis=1)

# Threshold value
SEUIL_DE_VALEUR_ELEVEE = 23

# Replace infinite values with NaN and drop rows with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Ensure 'X' (longitude) and 'Y' (latitude) are within valid range
df = df[(df['X'].between(-180, 180)) & (df['Y'].between(-90, 90))]

# Set the initial center of the map to your area of interest
center_latitude = 43.6045  # Replace with the latitude of your area
center_longitude = 1.4442  # Replace with the longitude of your area
initial_zoom_level = 9      # Adjust the zoom level to your preference

# Create the map with the specified center and zoom level
carte = Map(center=(center_latitude, center_longitude), zoom=initial_zoom_level)

# Add WMTS layer to the map
wmts_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS"
wmts_layer = TileLayer(url=wmts_url, name="WMTS Layer")
carte.add_layer(wmts_layer)
# Function to create heatmap layer
def create_heatmap_layer(df):
    # Extract the coordinates and pollution values as a list of tuples
    heatmap_data = [(row['Y'], row['X'], row['valeur']) for _, row in df.iterrows()]
    # Create the heatmap layer
    heatmap_layer = Heatmap(locations=heatmap_data, radius=20, blur=10, max_zoom=1)
    return heatmap_layer

def pollution_color(valeur):
    """Function to determine color based on pollution level."""
    if valeur > 50:
        return "red"
    elif valeur > 30:
        return "orange"
    elif valeur > 10:
        return "yellow"
    else:
        return "green"

def create_geojson_layer(df, threshold):
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    for _, row in df.iterrows():
        color = pollution_color(row['valeur'])
        feature = {
            "type": "Feature",
            "geometry": row['geometry'],
            "properties": row.to_dict(),
            "style": {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.8}
        }
        geojson_data['features'].append(feature)
    return geojson_data

# Create and add GeoJSON layer
geojson_layer = create_geojson_layer(df, SEUIL_DE_VALEUR_ELEVEE)
carte.add_layer(geojson_layer)

# Update the legend to reflect heatmap colors
legend = widgets.VBox([
    widgets.HTML(value="<b>Légende de la Heatmap</b>"),
    widgets.HTML(value='<div style="background:red; width:24px; height:24px; display:inline-block;"></div> Très Haute Intensité'),
    widgets.HTML(value='<div style="background:orange; width:24px; height:24px; display:inline-block;"></div> Haute Intensité'),
    widgets.HTML(value='<div style="background:yellow; width:24px; height:24px; display:inline-block;"></div> Intensité Moyenne'),
    widgets.HTML(value='<div style="background:green; width:24px; height:24px; display:inline-block;"></div> Basse Intensité')
])




# Ajouter les marqueurs à la carte

for marker in high_value_markers:
    carte.add_layer(marker)

# Display the map with the legend
display(widgets.HBox([carte, legend]))

# Create and add the heatmap layer
heatmap_layer = create_heatmap_layer(df)
carte.add_layer(heatmap_layer)


# %%
