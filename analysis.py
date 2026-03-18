import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load local CSV
df = pd.read_csv("owid-covid-data.csv")
print(f"Data loaded! Shape: {df.shape}")

# Clean the data
# Remove rows that are continents/world aggregates, keep only countries
df = df[df['continent'].notna()]

# Convert date column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Drop rows missing key columns we need
df = df.dropna(subset=['total_cases', 'total_deaths', 'people_fully_vaccinated_per_hundred'])

print(f"After cleaning: {df.shape}")
print(f"Countries included: {df['location'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

top_countries = ['United States', 'India', 'Brazil', 'France', 'Germany']
df_top = df[df['location'].isin(top_countries)]

plt.figure(figsize=(12,6))
for country in top_countries:
    data = df_top[df_top['location'] == country]
    plt.plot(data['date'], data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.tight_layout()
plt.savefig('chart1_case_trends.png')
plt.close()
print("Chart 1 saved!")

latest = df.groupby('location').last().reset_index()
top_deaths = latest.nlargest(15, 'total_deaths_per_million')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_deaths, x='total_deaths_per_million', y='location')
plt.title('Total COVID-19 Deaths per Million by Country')
plt.xlabel('Deaths per Million')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('chart2_deaths_per_million.png')
plt.close()
print("Chart 2 saved!")

latest_clean = latest.dropna(subset=['people_fully_vaccinated_per_hundred', 'total_deaths_per_million'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=latest_clean, x='people_fully_vaccinated_per_hundred', y='total_deaths_per_million', hue='continent')
plt.title('Vaccination Rate vs Deaths per Million by Country')
plt.xlabel('People Fully Vaccinated (%)')
plt.ylabel('Total Deaths per Million')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart3_vaccination_vs_mortality.png')
plt.close()
print("Chart 3 saved!")

df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

heatmap_data = df.groupby(['year', 'month'])['new_cases'].mean().unstack()

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=0.5, annot=False)
plt.title('Average New COVID-19 Cases by Month and Year')
plt.xlabel('Month')
plt.ylabel('Year')
plt.tight_layout()
plt.savefig('chart4_seasonal_heatmap.png')
plt.close()
print("Chart 4 saved!")

corr_data = latest.dropna(subset=['people_fully_vaccinated_per_hundred', 'total_deaths_per_million'])

correlation = corr_data['people_fully_vaccinated_per_hundred'].corr(corr_data['total_deaths_per_million'])

print(f"Correlation between vaccination rate and deaths per million: {correlation:.2f}")
print(f"Countries analyzed: {len(corr_data)}")

if correlation < 0:
    print("Result: Higher vaccination rates correlate with LOWER deaths per million")
else:
    print("Result: Positive correlation found")