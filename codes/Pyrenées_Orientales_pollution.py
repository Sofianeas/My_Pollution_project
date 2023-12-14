#%%
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime
from plotly.subplots import make_subplots
#%%
weather_file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/Donnees/Mesure_mensuelle_Region_Occitanie_Polluants_Principaux.csv'
df = pd.read_csv(weather_file_path, delimiter=',')
df.head()
#%%
# Describe the 'nom_dept' column
nom_dept_description = df['nom_dept'].describe()
nom_dept_description
df['nom_dept'].unique
#%%
# Filter the DataFrame for rows where 'nom_dept' is 'HERAULT'
df_PYRENEES_ORIENTALES= df[df['nom_dept'] == 'PYRENEES-ORIENTALES']

df_PYRENEES_ORIENTALES.head()

# %%
# Load the weather data
weather_file_path = 'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/PYRENEES_Orientales.csv'
weather_data = pd.read_csv(weather_file_path, delimiter=';')

# Convert 'date_debut' in df_HERAULT and 'Date' in weather_data to datetime
df_PYRENEES_ORIENTALES['date_debut'] = pd.to_datetime(df_PYRENEES_ORIENTALES['date_debut'], utc=True)
weather_data['Date'] = pd.to_datetime(weather_data['Date'], utc=True)

# Convert both indices to timezone-naive datetime (remove timezone)
df_PYRENEES_ORIENTALES['date_debut'] = df_PYRENEES_ORIENTALES['date_debut'].dt.tz_localize(None)
weather_data['Date'] = weather_data['Date'].dt.tz_localize(None)

# Set 'date_debut' and 'Date' as the indices
df_PYRENEES_ORIENTALES.set_index('date_debut', inplace=True)
weather_data.set_index('Date', inplace=True)

# Merge the datasets
combined_data = pd.merge(df_PYRENEES_ORIENTALES, weather_data, left_index=True, right_index=True, how='inner')

# Check the combined data
print(combined_data.head())

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
print(combined_data.head())
#%%
# Assuming df_HERAULT and weather_data are your original datasets

# Reset index of df_HERAULT if 'date_debut' is set as index
if isinstance(df_PYRENEES_ORIENTALES.index, pd.DatetimeIndex):
    df_PYRENEES_ORIENTALES = df_PYRENEES_ORIENTALES.reset_index()

# Now merge df_HERAULT with weather_data
# Assuming the weather data is already aggregated to the same time frequency (e.g., monthly)
combined_data = pd.merge(df_PYRENEES_ORIENTALES, weather_data, left_on='date_debut', right_index=True, how='inner')

# Check if 'date_debut' is now in the combined_data
print(combined_data.columns)

# Proceed with selecting your relevant columns
relevant_columns = ['date_debut', 'nom_poll', 'valeur', 'date_fin', 'Température', 'Humidité', 
                    'Vitesse du vent moyen 10 mn', 'nom_station', 'typologie', 'influence']
sub_dataframe = combined_data[relevant_columns]

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
def create_interactive_dual_axis_plot(x, y1, y2, y1_label, y2_label, y1_color, y2_color, title, filename):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1_label, mode='lines', line=dict(color=y1_color)),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x, y=y2, name=y2_label, mode='lines', line=dict(color=y2_color)),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=title,
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        paper_bgcolor='LightSteelBlue',
        plot_bgcolor='LightSteelBlue'
    )

    fig.update_yaxes(title_text=y1_label, secondary_y=False)
    fig.update_yaxes(title_text=y2_label, secondary_y=True)

    # Save the figure as an HTML file
    file_path = f'C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/docs/{filename}.html'

    fig.write_html(file_path)
    fig.show()

# Using the function to create and save plots
create_interactive_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Température'], sub_dataframe['valeur'],
                                  'Temperature (°C)', 'Pollution Value', 'red', 'purple',
                                  'Interactive Plot: Temperature vs Pollution Levels',
                                  'PYRENEES_ORIENTALES_Temperature_vs_Pollution_Levels')

create_interactive_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Humidité'], sub_dataframe['valeur'],
                                  'Humidity (%)', 'Pollution Value', 'blue', 'purple',
                                  'Interactive Plot: Humidity vs Pollution Levels',
                                  'PYRENEES_ORIENTALES_Humidity_vs_Pollution_Levels')

create_interactive_dual_axis_plot(sub_dataframe['date_debut'], sub_dataframe['Vitesse du vent moyen 10 mn'], sub_dataframe['valeur'],
                                  'Wind Speed (km/h)', 'Pollution Value', 'green', 'purple',
                                  'Interactive Plot: Wind Speed vs Pollution Levels',
                                  'PYRENEES_ORIENTALES_Wind_Speed_vs_Pollution_Levels')
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

#%%


# Assuming 'nom_poll' is the pollutant name and 'valeur' is its value
sub_dataframe.groupby('nom_poll')['valeur'].mean().plot(kind='bar')
plt.title('Average Distribution of Pollutants in Pyrenees Orientales')
plt.xlabel('Pollutant')
plt.ylabel('Average Value')
plt.show()

# %%
sub_dataframe.groupby('nom_poll')['valeur'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Different Pollutants in Pyrenees Orientales')
plt.ylabel('')  # Hide the y-label
plt.show()

# %%


# Example: Relationship between Temperature and Pollution Value
sns.scatterplot(data=sub_dataframe, x='Température', y='valeur')
plt.title('Temperature vs Pollution Value in Pyrenees Orientales')
plt.xlabel('Temperature')
plt.ylabel('Pollution Value')
plt.show()
# %%
sub_dataframe.resample('Y')['valeur'].mean().plot(kind='line')
plt.title('Yearly Trend of Pollution in Pyrenees Orientales')
plt.xlabel('Year')
plt.ylabel('Average Pollution Value')
plt.show()


# %%
sub_dataframe['valeur'].plot(kind='hist', bins=20)
plt.title('Frequency Distribution of Pollution Values in Hérault')
plt.xlabel('Pollution Value')
plt.ylabel('Frequency')
plt.show()

# %%
# sub_dataframe is already defined and contains the data for 'nom_poll' and 'valeur'
pollutant_concentration = sub_dataframe.groupby('nom_poll')['valeur'].sum().reset_index()

# Creating an interactive pie chart
fig = px.pie(pollutant_concentration, names='nom_poll', values='valeur', 
             title="Répartition des concentrations de pollution aux Pyrenées Orientales")

# Enhancements for a more stylish look
fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(
    title_font_size=20,
    showlegend=True,
    legend_title_text='Polluants',
    paper_bgcolor='LightSteelBlue',
    plot_bgcolor='LightSteelBlue',
    font=dict(size=15, color="black"),
)

# Display the figure
fig.show()
fig.write_html('C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/docs/Pyrenees_Orientales_pie_chart.html',full_html=False, include_plotlyjs='cdn')
# %%

    


# %%
# Convert 'date_debut' to month-year format for easier grouping
sub_dataframe['MonthYear'] = sub_dataframe['date_debut'].dt.to_period('M').dt.to_timestamp()

# Group by month-year and pollutant name, then sum the concentrations
monthly_concentration = sub_dataframe.groupby(['MonthYear', 'nom_poll'])['valeur'].sum().unstack()

# Create a subplot to stack bars
fig = make_subplots(rows=1, cols=1)

# Define colors for each pollutant - ensure there are enough colors for all pollutants
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']

# Create a bar for each pollutant
for idx, poll in enumerate(monthly_concentration.columns):
    fig.add_trace(
        go.Bar(
            x=monthly_concentration.index.strftime('%b %Y'),  # Formatting date for better readability
            y=monthly_concentration[poll],
            name=poll,
            marker_color=colors[idx % len(colors)]  # Rotate through colors if not enough
        )
    )

# Update layout for stacked bar chart
fig.update_layout(
    title="Concentration maximale de pollution aux Pyrenées Orientales par mois",
    xaxis_title="Mois",
    yaxis_title="Concentration de pollution (µg/m³)",
    barmode='stack',
    paper_bgcolor='LightSteelBlue',
    plot_bgcolor='LightSteelBlue',
    font=dict(size=15, color="black"),
)

# Add legend to the right side
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

# Display the figure
fig.show()
fig.write_html('C:/Users/SCD-UM/OneDrive/Bureau/Project/Projet_Groupe_Pollution_Air_Occitanie/docs/Pyrenées_Orientales_bar_chart.html',full_html=False, include_plotlyjs='cdn')
# %%
