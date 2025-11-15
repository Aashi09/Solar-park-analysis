# solar_parks_analysis.py
# Complete Solar Parks Evaluation Project in One File

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def load_data():
    print("Loading datasets...")
    parks = pd.read_csv('Solar-park-analysis\datasets\india_solar_parks_dataset.csv')
    policy = pd.read_csv('Solar-park-analysis\datasets\solar_parks_policy_timeline.csv')
    challenges = pd.read_csv('Solar-park-analysis/datasets/solar_parks_challenges.csv')
    benchmarks = pd.read_csv('Solar-park-analysis\datasets\solar_parks_benchmarks.csv')
    print("Datasets loaded successfully.\n")
    return parks, policy, challenges, benchmarks
def exploratory_analysis(parks):
    print("=== Exploratory Data Analysis ===")
    print("\nDataset Overview:\n", parks.head())
    print("\nSummary statistics for key metrics:\n", parks[['planned_capacity_mw', 'operational_capacity_mw', 'performance_ratio_percent']].describe())
    parks['land_efficiency_mw_per_acre'] = parks['operational_capacity_mw'] / parks['land_area_acres']
    print("\nLand Efficiency (MW per Acre) for parks:\n", parks[['park_name', 'land_efficiency_mw_per_acre']])

def plot_top_parks_capacity(parks):
    top5 = parks.sort_values('operational_capacity_mw', ascending=False).head(5)
    plt.figure(figsize=(10,5))
    plt.barh(top5['park_name'], top5['operational_capacity_mw'], color='royalblue')
    plt.xlabel('Operational Capacity (MW)')
    plt.title('Top 5 Solar Parks by Operational Capacity')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_performance_vs_capacity(parks):
    plt.figure(figsize=(8,5))
    plt.scatter(parks['operational_capacity_mw'], parks['performance_ratio_percent'], color='orange')
    plt.xlabel('Operational Capacity (MW)')
    plt.ylabel('Performance Ratio (%)')
    plt.title('Performance Ratio vs Capacity for Solar Parks')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_challenges_impact(challenges):
    challenges_sorted = challenges.sort_values('impact_score', ascending=False)
    plt.figure(figsize=(10,6))
    plt.barh(challenges_sorted['challenge'], challenges_sorted['impact_score'], color='tomato')
    plt.xlabel('Impact Score')
    plt.title('Major Challenges in Solar Park Development')
    plt.tight_layout()
    plt.show()

def plot_state_capacity(parks):
    state_caps = parks.groupby('state')['operational_capacity_mw'].sum().sort_values(ascending=True)
    plt.figure(figsize=(10,5))
    state_caps.plot(kind='barh', color='seagreen')
    plt.xlabel('Total Operational Capacity (MW)')
    plt.title('Operational Capacity by State')
    plt.tight_layout()
    plt.show()

def plot_policy_trends(policy):
    plt.figure(figsize=(8,5))
    plt.plot(policy['year'], policy['cumulative_operational_mw'], marker='o', color='purple')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Operational MW')
    plt.title('Growth Trend of Solar Parks Capacity Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_effectiveness_score(parks):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(parks[['performance_ratio_percent', 'capacity_utilization_factor_percent']].fillna(0))
    parks['pr_scaled'] = scaled[:,0]
    parks['cuf_scaled'] = scaled[:,1]
    parks['effectiveness_score'] = 0.6 * parks['pr_scaled'] + 0.4 * parks['cuf_scaled']
    print("\nParks sorted by Effectiveness Score:")
    print(parks[['park_name', 'performance_ratio_percent', 'capacity_utilization_factor_percent', 'effectiveness_score']].sort_values('effectiveness_score', ascending=False).to_string(index=False))

def main():
    parks, policy, challenges, benchmarks = load_data()
    exploratory_analysis(parks)
    plot_top_parks_capacity(parks)
    plot_performance_vs_capacity(parks)
    plot_challenges_impact(challenges)
    plot_state_capacity(parks)
    plot_policy_trends(policy)
    calculate_effectiveness_score(parks)
    print("\n--- Project completed successfully! ---\n")

if __name__ == "__main__":
    main()