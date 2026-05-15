import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import imageio

class AirQualityPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.filtered_data = None
        self.stats = {}
        
        # Ensure directories exist for GitHub/Grading compliance
        for folder in ['outputs', 'data']:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def ingest_data(self):
        """Step 1: Data Ingestion with Error Handling"""
        try:
            self.raw_data = pd.read_csv(self.file_path)
            self.raw_data['Timestamp'] = pd.to_datetime(self.raw_data['Timestamp'])
            print("Step 1: Data ingestion successful.")
        except Exception as e:
            print(f"Ingestion Error: {e}")

    def clean_and_filter(self):
        """Step 2: Unique Filter Logic (Summer Months: 3, 4, 5)"""
        try:
            df = self.raw_data.drop_duplicates().dropna()
            # Filter for March, April, and May
            self.filtered_data = df[df['Month'].isin([3, 4, 5])].copy()
            self.filtered_data.to_csv('data/dataset_cleaned.csv', index=False)
            print(f"Step 2: Filtered to {len(self.filtered_data)} summer observations.")
        except Exception as e:
            print(f"Filtering Error: {e}")

    def run_numpy_analytics(self):
        """Mandatory NumPy Integration for Engineering Stats"""
        if self.filtered_data is None: return
        values = self.filtered_data['PM2.5'].values
        
        self.stats = {
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std Dev': np.std(values),
            'Variance': np.var(values),
            'Max': np.max(values),
            'Min': np.min(values)
        }
        
        print("\n--- IEEE SECTION V: NUMPY ANALYTICS ---")
        for metric, value in self.stats.items():
            print(f"{metric}: {value:.4f}")

    def create_visualizations(self):
        """Step 3: 6 Static & 3 Animated Visualizations"""
        if self.filtered_data is None: return
        
        # --- 6 STATIC GRAPHS ---
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=self.filtered_data, x='Hour', y='PM2.5')
        plt.title("Static 1: Diurnal PM2.5 Cycles (Summer)")
        plt.savefig('outputs/static_1_hourly_trend.png')
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=self.filtered_data['PM2.5'], color='skyblue')
        plt.title("Static 2: PM2.5 Concentration Spread")
        plt.savefig('outputs/static_2_boxplot.png')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.filtered_data.corr(), annot=True, cmap='YlGnBu')
        plt.title("Static 3: Correlation Heatmap")
        plt.savefig('outputs/static_3_heatmap.png')

        plt.figure(figsize=(8, 5))
        sns.violinplot(x='Month', y='PM2.5', data=self.filtered_data)
        plt.title("Static 4: Density Distribution by Month")
        plt.savefig('outputs/static_4_violin.png')

        plt.figure(figsize=(8, 5))
        sns.ecdfplot(data=self.filtered_data, x='PM2.5', hue='Month')
        plt.title("Static 5: Cumulative Distribution Function")
        plt.savefig('outputs/static_5_cdf.png')

        plt.figure(figsize=(8, 5))
        sns.barplot(data=self.filtered_data, x='Month', y='PM2.5', estimator=np.mean)
        plt.title("Static 6: Monthly Average Intensity")
        plt.savefig('outputs/static_6_bar.png')
        
        # --- 3 ANIMATED GRAPHS (HTML) ---
        fig1 = px.histogram(self.filtered_data, x="PM2.5", color="Month",
                           animation_frame="Hour", title="Animated 1: Hourly Distribution")
        fig1.write_html("outputs/animated_1_dist.html")

        fig2 = px.scatter(self.filtered_data, x="Day", y="PM2.5", color="Hour",
                          animation_frame="Month", title="Animated 2: Daily Variations")
        fig2.write_html("outputs/animated_2_scatter.html")

        avg_path = self.filtered_data.groupby(['Hour', 'Month'])['PM2.5'].mean().reset_index()
        fig3 = px.line(avg_path, x="Hour", y="PM2.5", color="Month",
                       animation_frame="Hour", title="Animated 3: Diurnal Path")
        fig3.write_html("outputs/animated_3_path.html")
        
        print("\nStep 3: 9 Visualizations saved to /outputs folder.")

    def export_animated_gif(self):
        """BONUS: Converting the Diurnal Path to GIF for GitHub README"""
        print("\nStep 4: Generating GIF frames (this may take a minute)...")
        avg_path = self.filtered_data.groupby(['Hour', 'Month'])['PM2.5'].mean().reset_index()
        frames = []
        
        for h in sorted(avg_path['Hour'].unique()):
            current_data = avg_path[avg_path['Hour'] <= h]
            fig = px.line(current_data, x="Hour", y="PM2.5", color="Month",
                          range_x=[0, 23], range_y=[0, avg_path['PM2.5'].max() + 10],
                          title=f"Diurnal PM2.5 Cycle: Hour {h}")
            img_bytes = fig.to_image(format="png", engine="kaleido")
            frames.append(imageio.v2.imread(img_bytes))
            
        imageio.mimsave('outputs/diurnal_cycle_animation.gif', frames, fps=4)
        print("Step 4: GIF successfully saved to /outputs/diurnal_cycle_animation.gif")

if __name__ == "__main__":
    # Ensure this matches your file in the /data folder
    path = 'data/air-quality-india.csv'
    
    pipeline = AirQualityPipeline(path)
    pipeline.ingest_data()
    pipeline.clean_and_filter()
    pipeline.run_numpy_analytics()
    pipeline.create_visualizations()
    
    # Run GIF export (Requires: pip install kaleido imageio)
    try:
        pipeline.export_animated_gif()
    except Exception as e:
        print(f"GIF Export skipped: {e}. (Tip: pip install kaleido imageio)")