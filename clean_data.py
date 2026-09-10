"""
Tech Job Market & Salary Trends Analyzer
Script: clean_data.py
Description: Day 2 - Data Cleaning, Categorical Standardization, and Outlier Handling.
Dependencies: pandas, numpy
"""

import os
import numpy as np
import pandas as pd


def clean_salary_dataset():
    """
    Loads raw salary data, selects essential dashboard columns, removes invalid/duplicate
    records, standardizes categorical codes, filters realistic salary outliers,
    and exports a pristine cleaned CSV file.
    """
    print("=" * 75)
    print(" DAY 2: DATA CLEANING, STANDARDIZATION & OUTLIER HANDLING ")
    print(" Project: Tech Job Market & Salary Trends Analyzer")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STEP 1: Load the Raw Dataset with Error Handling
    # -------------------------------------------------------------------------
    raw_path = os.path.join("data", "ai_ds_job_salaries_2026.csv")
    output_path = os.path.join("data", "cleaned_salaries.csv")

    print(f"\n[STEP 1] Loading raw dataset from: '{raw_path}'...")

    if not os.path.exists(raw_path):
        print(f"[ERROR] Source file '{raw_path}' not found!")
        print("Please ensure 'ai_ds_job_salaries_2026.csv' exists inside the 'data/' folder.")
        return

    df_raw = pd.read_csv(raw_path)
    initial_rows, initial_cols = df_raw.shape
    print(f" -> Raw dataset loaded successfully: {initial_rows:,} rows, {initial_cols} columns.")

    # -------------------------------------------------------------------------
    # STEP 2: Detect & Select Essential Columns for the Dashboard
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Detecting and mapping essential columns...")

    # Define aliases/variations for common column names across salary datasets
    column_aliases = {
        "job_title": ["job_title", "title", "role", "job_role", "position"],
        "experience_level": ["experience_level", "exp_level", "experience", "seniority"],
        "salary_usd": ["salary_usd", "salary_in_usd", "salary", "annual_salary_usd", "salary_in_dollars"],
        "employment_type": ["employment_type", "emp_type", "employment_status", "work_type"],
        "remote_ratio": ["remote_ratio", "remote_work_ratio", "remote_percentage", "work_setting"],
        "company_location": ["company_location", "company_country", "location", "country"],
        "year": ["work_year", "year", "posting_year", "survey_year", "report_year"],
        # Optional high-value context columns for skills & market segmentation
        "primary_language": ["primary_language", "primary_tech", "skills", "tech_stack", "programming_language"],
        "company_size": ["company_size", "size", "org_size"],
        "industry": ["industry", "company_industry", "sector"],
        "years_experience": ["years_experience", "experience_years", "yo_exp"]
    }

    selected_cols = {}
    for standard_name, aliases in column_aliases.items():
        for col in df_raw.columns:
            if col.lower() in aliases:
                selected_cols[standard_name] = col
                break

    # Validate that mandatory core columns exist
    mandatory_fields = ["job_title", "experience_level", "salary_usd", "employment_type", "remote_ratio", "company_location"]
    missing_mandatory = [f for f in mandatory_fields if f not in selected_cols]

    if missing_mandatory:
        print(f"[ERROR] Could not automatically identify required column(s): {missing_mandatory}")
        return

    # Create a copy with standardized column names
    rename_mapping = {raw_col: std_name for std_name, raw_col in selected_cols.items()}
    df = df_raw[list(rename_mapping.keys())].rename(columns=rename_mapping).copy()

    # If dataset lacks an explicit 'year' column, default to 2026 (the dataset survey period)
    if "year" not in df.columns:
        df["year"] = 2026
        print(" -> 'year' column not explicitly in raw data; populated with default survey year (2026).")

    print(f" -> Retained columns for dashboard: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # STEP 3: Remove Duplicates & Null/Non-Positive Salaries
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Dropping duplicates and invalid records...")

    # 3.1 Drop exact duplicate rows (can distort sample weights and statistical models)
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f" -> Removed {duplicates_count:,} duplicate row(s).")
    else:
        print(" -> No duplicate rows found.")

    # 3.2 Ensure salary is numeric, non-null, and strictly positive (> $0)
    df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")
    invalid_salary_mask = df["salary_usd"].isna() | (df["salary_usd"] <= 0)
    invalid_salary_count = invalid_salary_mask.sum()

    if invalid_salary_count > 0:
        df = df[~invalid_salary_mask]
        print(f" -> Removed {invalid_salary_count:,} row(s) with missing or non-positive salaries.")
    else:
        print(" -> All salary values are valid positive numbers.")

    # -------------------------------------------------------------------------
    # STEP 4: Standardize Categorical Columns
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Standardizing categorical variables...")

    # 4.1 Experience Level Standardization:
    # Standardizes abbreviations (EN, MI, SE, EX) or single-word levels to clear, full labels.
    experience_mapping = {
        # Abbreviations
        "EN": "Entry-Level",
        "MI": "Mid-Level",
        "SE": "Senior-Level",
        "EX": "Executive/Lead",
        # Full / informal words
        "Entry": "Entry-Level",
        "entry": "Entry-Level",
        "Entry-level": "Entry-Level",
        "Entry-Level": "Entry-Level",
        "Junior": "Entry-Level",
        "Mid": "Mid-Level",
        "mid": "Mid-Level",
        "Mid-level": "Mid-Level",
        "Mid-Level": "Mid-Level",
        "Intermediate": "Mid-Level",
        "Senior": "Senior-Level",
        "senior": "Senior-Level",
        "Senior-level": "Senior-Level",
        "Senior-Level": "Senior-Level",
        "Lead": "Executive/Lead",
        "lead": "Executive/Lead",
        "Executive": "Executive/Lead",
        "executive": "Executive/Lead",
        "Executive/Lead": "Executive/Lead",
        "Principal": "Executive/Lead",
        "Director": "Executive/Lead"
    }
    df["experience_level"] = df["experience_level"].astype(str).str.strip().map(experience_mapping).fillna(df["experience_level"])
    print(" -> Standardized 'experience_level' into: Entry-Level, Mid-Level, Senior-Level, Executive/Lead.")

    # 4.2 Remote Ratio Standardization:
    # Converts 0%, 50%, 100% or text into clear, user-friendly labels ('On-site', 'Hybrid', 'Remote').
    remote_mapping = {
        0: "On-site", 0.0: "On-site", "0": "On-site", "0%": "On-site", "On-site": "On-site", "Onsite": "On-site",
        50: "Hybrid", 50.0: "Hybrid", "50": "Hybrid", "50%": "Hybrid", "Hybrid": "Hybrid",
        100: "Remote", 100.0: "Remote", "100": "Remote", "100%": "Remote", "Remote": "Remote", "Fully Remote": "Remote"
    }
    df["remote_ratio"] = df["remote_ratio"].map(remote_mapping).fillna(df["remote_ratio"].astype(str))
    print(" -> Standardized 'remote_ratio' into: On-site, Hybrid, Remote.")

    # 4.3 Clean Text & Title Casing for Job Titles and Employment Types
    df["job_title"] = df["job_title"].astype(str).str.strip()
    df["employment_type"] = df["employment_type"].astype(str).str.strip().str.title()
    df["company_location"] = df["company_location"].astype(str).str.strip().str.upper()

    # -------------------------------------------------------------------------
    # STEP 5: Outlier Handling & Salary Filtering
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Filtering extreme salary outliers...")

    # In tech compensation analysis, salaries under $15,000/year (often unpaid/part-time artifacts)
    # or exceeding $500,000/year (extreme data entry errors or skewed executive packages)
    # can heavily distort dashboard averages and visualizations.
    min_salary_threshold = 15000
    max_salary_threshold = 500000

    outliers_mask = (df["salary_usd"] < min_salary_threshold) | (df["salary_usd"] > max_salary_threshold)
    outliers_count = outliers_mask.sum()

    if outliers_count > 0:
        df = df[~outliers_mask].copy()
        print(f" -> Removed {outliers_count:,} salary outlier row(s) outside [${min_salary_threshold:,.0f} - ${max_salary_threshold:,.0f}].")
    else:
        print(" -> No salary values outside domain thresholds detected.")

    # -------------------------------------------------------------------------
    # STEP 6: Save Cleaned Data to CSV
    # -------------------------------------------------------------------------
    print(f"\n[STEP 6] Saving cleaned dataset to: '{output_path}'...")
    # index=False prevents pandas from writing row numbers as an unnecessary extra column
    df.to_csv(output_path, index=False)
    print(f" -> [SUCCESS] File saved successfully ({os.path.getsize(output_path):,} bytes).")

    # -------------------------------------------------------------------------
    # STEP 7: Comprehensive Summary Report
    # -------------------------------------------------------------------------
    final_rows, final_cols = df.shape
    total_removed = initial_rows - final_rows
    pct_retained = (final_rows / initial_rows) * 100

    print("\n" + "=" * 75)
    print(" DATA CLEANING & STANDARDIZATION SUMMARY REPORT ")
    print("=" * 75)

    print("\n1. RECORD COUNTS:")
    print(f"  * Raw Data Rows:       {initial_rows:,}")
    print(f"  * Cleaned Data Rows:   {final_rows:,}")
    print(f"  * Total Rows Removed:  {total_removed:,} ({100 - pct_retained:.2f}%)")
    print(f"  * Data Retention Rate: {pct_retained:.2f}%")

    print("\n2. EXPERIENCE LEVEL DISTRIBUTION (Cleaned):")
    exp_counts = df["experience_level"].value_counts()
    for level, count in exp_counts.items():
        pct = (count / final_rows) * 100
        bar = "#" * int(pct // 2)
        print(f"  * {level:<16} : {count:>5,} postings ({pct:>5.1f}%) | {bar}")

    print("\n3. REMOTE WORK RATIO DISTRIBUTION (Cleaned):")
    remote_counts = df["remote_ratio"].value_counts()
    for mode, count in remote_counts.items():
        pct = (count / final_rows) * 100
        bar = "#" * int(pct // 2)
        print(f"  * {mode:<16} : {count:>5,} postings ({pct:>5.1f}%) | {bar}")

    print("\n4. SALARY METRICS (USD - Cleaned):")
    sal = df["salary_usd"]
    print(f"  * Minimum Salary:        ${sal.min():>10,.2f}")
    print(f"  * 25th Percentile (Q1):  ${sal.quantile(0.25):>10,.2f}")
    print(f"  * Median Salary (50%):   ${sal.median():>10,.2f}")
    print(f"  * Mean (Average) Salary: ${sal.mean():>10,.2f}")
    print(f"  * 75th Percentile (Q3):  ${sal.quantile(0.75):>10,.2f}")
    print(f"  * Maximum Salary:        ${sal.max():>10,.2f}")
    print(f"  * Standard Deviation:    ${sal.std():>10,.2f}")

    print("\n5. TOP 5 JOB TITLES IN CLEANED DATASET:")
    top_titles = df["job_title"].value_counts().head(5)
    for rank, (title, count) in enumerate(top_titles.items(), 1):
        print(f"  {rank}. {title:<30} -> {count:>4,} postings ({(count/final_rows)*100:>4.1f}%)")

    print("\n" + "=" * 75)
    print(" Day 2 Cleaning Pipeline Complete! Ready for Day 3 EDA & Dashboard. ")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    clean_salary_dataset()
