"""
Tech Job Market & Salary Trends Analyzer
Script: inspect_data.py
Description: Initial data exploration and sanity check for AI/DS job market dataset.
"""

import os
import pandas as pd

# ---------------------------------------------------------
# Step 1: Load the Dataset with Error Handling
# ---------------------------------------------------------
# Relative path to our CSV file inside the 'data' folder.
csv_file_path = os.path.join("data", "ai_ds_job_salaries_2026.csv")

print("=" * 70)
print("TECH JOB MARKET & SALARY TRENDS - INITIAL DATA INSPECTION")
print("=" * 70)

try:
    # pd.read_csv() reads a CSV file into a pandas DataFrame (2D tabular structure).
    print(f"Loading data from: {csv_file_path} ...")
    df = pd.read_csv(csv_file_path)
    print("[SUCCESS] Dataset successfully loaded into a pandas DataFrame!\n")
except FileNotFoundError:
    print(f"[ERROR] The file '{csv_file_path}' was not found.")
    print("Make sure you are running the script from the project root folder")
    print("and that 'ai_ds_job_salaries_2026.csv' exists inside the 'data/' directory.")
    exit(1)
except Exception as e:
    print(f"[ERROR] An unexpected error occurred while reading the file: {e}")
    exit(1)

# ---------------------------------------------------------
# Step 2: Dataset Dimensions (Rows & Columns)
# ---------------------------------------------------------
# df.shape returns a tuple: (number of rows, number of columns).
num_rows, num_cols = df.shape
print("-" * 70)
print("1. DATASET DIMENSIONS")
print("-" * 70)
print(f"Total Rows (Job Postings):    {num_rows:,}")
print(f"Total Columns (Features):    {num_cols}")
print()

# ---------------------------------------------------------
# Step 3: Preview the First 5 Rows
# ---------------------------------------------------------
# df.head(n) returns the first n rows of the DataFrame (default is 5).
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("-" * 70)
print("2. PREVIEW: FIRST 5 ROWS (.head())")
print("-" * 70)
print(df.head())
print()

# ---------------------------------------------------------
# Step 4: Inspect Column Types and Missing / Null Values
# ---------------------------------------------------------
# df.info() displays column names, non-null counts, and memory/data types.
print("-" * 70)
print("3. DATA STRUCTURE & MISSING VALUES (.info())")
print("-" * 70)
df.info()
print()

null_counts = df.isnull().sum()
total_nulls = null_counts.sum()
print(f"Total missing values across entire dataset: {total_nulls}")
if total_nulls > 0:
    print("Columns with missing values:")
    print(null_counts[null_counts > 0])
else:
    print("Dataset Quality: No null or missing values detected!")
print()

# ---------------------------------------------------------
# Step 5: Dynamically Identify Salary Column & Compute Statistics
# ---------------------------------------------------------
# We dynamically search for the salary column by matching common naming patterns.
possible_salary_cols = ["salary_usd", "salary_in_usd", "salary", "annual_salary", "base_salary"]
salary_col = None

for col in df.columns:
    if col.lower() in possible_salary_cols:
        salary_col = col
        break

if salary_col is None:
    for col in df.columns:
        if "salary" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            salary_col = col
            break

print("-" * 70)
print("4. SALARY DISTRIBUTION & SUMMARY STATISTICS")
print("-" * 70)

if salary_col:
    print(f"Identified Salary Column: '{salary_col}'\n")
    
    # Statistical calculations using pandas methods:
    # .min() -> Minimum salary in the dataset
    # .quantile(0.25) -> 25th percentile (25% earn less than this)
    # .median() -> 50th percentile / middle value (robust against extreme outliers)
    # .mean() -> Arithmetic average salary
    # .quantile(0.75) -> 75th percentile (75% earn less than this)
    # .max() -> Highest salary in the dataset
    salary_min = df[salary_col].min()
    salary_q25 = df[salary_col].quantile(0.25)
    salary_median = df[salary_col].median()
    salary_mean = df[salary_col].mean()
    salary_q75 = df[salary_col].quantile(0.75)
    salary_max = df[salary_col].max()
    salary_std = df[salary_col].std()

    print(f"  * Minimum Salary:       ${salary_min:,.2f}")
    print(f"  * 25th Percentile (Q1): ${salary_q25:,.2f}")
    print(f"  * Median Salary (50%):  ${salary_median:,.2f}")
    print(f"  * Mean (Average):       ${salary_mean:,.2f}")
    print(f"  * 75th Percentile (Q3): ${salary_q75:,.2f}")
    print(f"  * Maximum Salary:       ${salary_max:,.2f}")
    print(f"  * Standard Deviation:   ${salary_std:,.2f}")
else:
    print("[WARNING] Could not automatically detect a numeric salary column.")
print()

# ---------------------------------------------------------
# Step 6: Top 5 Job Titles & Experience Level Breakdown
# ---------------------------------------------------------
# .value_counts() counts the frequency of each unique value in a categorical column.
print("-" * 70)
print("5. TOP 5 MOST COMMON JOB TITLES")
print("-" * 70)
if "job_title" in df.columns:
    top_jobs = df["job_title"].value_counts().head(5)
    for rank, (title, count) in enumerate(top_jobs.items(), 1):
        pct = (count / num_rows) * 100
        print(f"  {rank}. {title:<32} -> {count:>4} postings ({pct:>5.1f}%)")
else:
    print("Column 'job_title' not found.")
print()

print("-" * 70)
print("6. BREAKDOWN OF EXPERIENCE LEVELS")
print("-" * 70)
if "experience_level" in df.columns:
    exp_counts = df["experience_level"].value_counts()
    for level, count in exp_counts.items():
        pct = (count / num_rows) * 100
        print(f"  - {level:<15} : {count:>4} postings ({pct:>5.1f}%)")
else:
    print("Column 'experience_level' not found.")
print()

print("=" * 70)
print("Inspection complete! Ready for exploratory data analysis (EDA).")
print("=" * 70)
