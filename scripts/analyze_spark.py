import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qa_assessments.db')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ATTRIBUTES = ["resolution", "card", "face", "blur", "glare", "noise", "exposure", "geometry"]

def run_analysis():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}.")
        return

    conn = sqlite3.connect(DB_PATH)
    assessments_df = pd.read_sql_query("SELECT * FROM assessments", conn)
    feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()

    if assessments_df.empty:
        print("No data found.")
        return

    print("--- Starting Superset-style Analysis ---")
    
    # ── 1. Calculate Attribute Accuracy ──
    attribute_stats = []
    total = len(assessments_df)
    
    # Pre-build a matrix for heatmap
    error_matrix = pd.DataFrame(0, index=assessments_df.filename, columns=ATTRIBUTES)

    for attr in ATTRIBUTES:
        attr_feedback = feedback_df[(feedback_df['attribute'] == attr) & (feedback_df['is_wrong'] == 1)]
        errors = len(attr_feedback)
        accuracy = ((total - errors) / total) * 100 if total > 0 else 0
        
        attribute_stats.append({
            "Attribute": attr.capitalize(),
            "Accuracy (%)": accuracy
        })
        
        # Populate error matrix
        for _, row in attr_feedback.iterrows():
            filename = assessments_df.loc[assessments_df['id'] == row['assessment_id'], 'filename'].values[0]
            error_matrix.at[filename, attr] = 1

    stats_df = pd.DataFrame(attribute_stats)
    print(stats_df.to_string(index=False))

    # ── PySpark Showcase ──
    if HAS_PYSPARK:
        spark = SparkSession.builder.appName("IDCard_Superset").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        s_assess = spark.createDataFrame(assessments_df)
        print("\n[PySpark] Acceptance Distribution:")
        s_assess.groupBy("decision").count().show()
        spark.stop()

    # ── Generating Visualizations (Mimicking Superset Dashboards) ──
    plt.style.use('dark_background') # Superset dark mode aesthetic
    
    # 1. Bar Chart: Accuracy by Attribute
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(stats_df))
    bars = plt.bar(stats_df['Attribute'], stats_df['Accuracy (%)'], color=colors)
    plt.title('Attribute Validation Accuracy', fontsize=16, pad=20)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_by_attribute.png'))
    plt.close()

    # 2. Pie Chart: Decisions
    plt.figure(figsize=(8, 8))
    decision_counts = assessments_df['decision'].value_counts()
    plt.pie(decision_counts, labels=decision_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140, explode=[0.05]*len(decision_counts))
    plt.title('Accept vs Reject Distribution', fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, 'decision_distribution.png'))
    plt.close()

    # 3. Histogram: Quality Score
    plt.figure(figsize=(10, 6))
    sns.histplot(assessments_df['quality_score'], bins=10, kde=True, color='#3498db')
    plt.title('Aggregate Quality Score Distribution', fontsize=16, pad=20)
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'quality_score_distribution.png'))
    plt.close()

    # 4. Heatmap: Errors per file
    plt.figure(figsize=(12, 8))
    sns.heatmap(error_matrix, cmap='rocket_r', cbar=False, linewidths=.5)
    plt.title('Error Heatmap by File (1 = Error)', fontsize=16, pad=20)
    plt.ylabel('Filename (Timestamp)')
    plt.xlabel('Attributes')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_heatmap.png'))
    plt.close()

    # 5. Line Chart: Quality Trend
    plt.figure(figsize=(12, 5))
    plt.plot(assessments_df['filename'].str.extract(r'at (.*)')[0], assessments_df['quality_score'], marker='o', linestyle='-', color='#f1c40f', linewidth=2)
    plt.title('Quality Score Trend Over Uploads', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Quality Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'quality_score_trend.png'))
    plt.close()

    print(f"\nSuccessfully generated 5 Superset-style visualizations in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()
