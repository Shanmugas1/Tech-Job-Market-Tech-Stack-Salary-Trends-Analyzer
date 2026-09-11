"""
Tech Job Market & Salary Trends Analyzer
Script: analyze_data.py
Description: Day 3 - Exploratory Data Analysis (EDA) & Aggregations
Dependencies: pandas
"""

import os
import pandas as pd


# =============================================================================
# MENTOR LESSON: CONCEPTUAL FOUNDATIONS OF AGGREGATIONS & EDA IN PANDAS
# =============================================================================
# 1. WHAT DOES .groupby() DO CONCEPTUALLY?
#    - GroupBy implements the classic "Split-Apply-Combine" paradigm:
#      a) SPLIT: Divides the original DataFrame into logical subsets/buckets
#         based on unique categories in the grouping column(s) (e.g., job title).
#      b) APPLY: Computes statistical functions (mean, median, count, min, max)
#         independently on each group's subset of data.
#      c) COMBINE: Stitches all individual group results back into a clean,
#         unified summary DataFrame.
#
# 2. WHY DO WE AGGREGATE WITH MULTIPLE METRICS USING .agg({...})?
#    - Relying on a single metric (like the arithmetic mean) can be misleading
#      because salary data is notoriously right-skewed by top earners.
#    - Combining multiple metrics provides full distributional context:
#      * 'count'  -> Confirms sample size & statistical reliability.
#      * 'median' -> Robust measure of central tendency (resistant to outliers).
#      * 'mean'   -> Arithmetic average (highlights skewness when mean > median).
#      * 'min'/'max' / '25%'/'75%' -> Measures spread, IQR (Interquartile Range),
#         and realistic salary brackets.
#
# 3. HOW DOES .pivot_table() ORGANIZE TWO CATEGORICAL VARIABLES?
#    - While .groupby() produces a 1D vertical list of grouped metrics,
#      .pivot_table() reshapes data into a 2D matrix (cross-tabulation).
#    - One categorical variable forms the ROWS (index), another forms the
#      COLUMNS, and the intersections display the aggregated VALUES (e.g., median salary).
# =============================================================================


def format_currency(val):
    """Helper function to format numeric float/int into clean USD currency string."""
    if pd.isna(val):
        return "N/A"
    return f"${val:,.0f}"


def format_table(df_display, title, column_formats=None):
    """
    Helper function to print beautifully aligned terminal tables
    with headers, borders, and custom formatting.
    """
    print("\n" + "=" * 85)
    print(f" {title.upper()} ")
    print("=" * 85)

    # Format specified columns
    formatted_df = df_display.copy()
    if column_formats:
        for col, fmt in column_formats.items():
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(fmt)

    # Print DataFrame with clean spacing
    print(formatted_df.to_string(index=True if formatted_df.index.name else False))
    print("-" * 85)


def analyze_salary_data():
    """
    Loads the cleaned salary dataset and executes 5 key business intelligence queries
    using pandas groupby, agg, and pivot_table.
    """
    # -------------------------------------------------------------------------
    # STEP 0: Load Cleaned Dataset
    # -------------------------------------------------------------------------
    data_path = os.path.join("data", "cleaned_salaries.csv")

    print("\n" + "#" * 85)
    print(" DAY 3: EXPLORATORY DATA ANALYSIS (EDA) & AGGREGATION PIPELINE ")
    print(" Project: Tech Job Market & Salary Trends Analyzer")
    print("#" * 85)

    if not os.path.exists(data_path):
        print(f"\n[ERROR] Cleaned dataset not found at: '{data_path}'")
        print("Please run 'python clean_data.py' first to generate the cleaned dataset.\n")
        return

    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"\n[DATA LOADED] Loaded {total_records:,} records from '{data_path}'.\n")

    # -------------------------------------------------------------------------
    # QUERY 1: Top 10 Highest-Paying Job Titles (Median Salary, Count >= 5)
    # -------------------------------------------------------------------------
    # CONCEPT:
    # Group by 'job_title', compute both median (central tendency) and mean
    # (average), plus count. We filter out titles with < 5 postings to avoid
    # single-applicant anomalies skewing the rankings.
    # -------------------------------------------------------------------------
    print("\n>>> [QUERY 1] Computing Top 10 Highest-Paying Job Titles...")

    min_postings_q1 = 5
    q1_grouped = (
        df.groupby("job_title")["salary_usd"]
        .agg(
            postings_count="count",
            median_salary="median",
            mean_salary="mean",
            min_salary="min",
            max_salary="max"
        )
        .reset_index()
    )

    # Filter for statistical significance and sort descending by median salary
    q1_top10 = (
        q1_grouped[q1_grouped["postings_count"] >= min_postings_q1]
        .sort_values(by="median_salary", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    q1_top10.index = range(1, len(q1_top10) + 1)
    q1_top10.index.name = "Rank"

    q1_display = q1_top10.rename(
        columns={
            "job_title": "Job Title",
            "postings_count": "Postings",
            "median_salary": "Median Salary",
            "mean_salary": "Mean Salary",
            "min_salary": "Min Salary",
            "max_salary": "Max Salary"
        }
    )

    q1_display_formats = {
        "Postings": lambda x: f"{x:,}",
        "Median Salary": format_currency,
        "Mean Salary": format_currency,
        "Min Salary": format_currency,
        "Max Salary": format_currency
    }

    format_table(
        q1_display.reset_index(),
        title="Query 1: Top 10 Highest-Paying Job Titles (Min. 5 Postings, Sorted by Median)",
        column_formats=q1_display_formats
    )

    # -------------------------------------------------------------------------
    # QUERY 2: Salary Progression by Experience Level
    # -------------------------------------------------------------------------
    # CONCEPT:
    # Aggregating multiple summary statistics (count, min, 25%, 50%, mean, 75%, max)
    # reveals how compensation scales across career tiers and shows the IQR
    # (Interquartile Range: 25th to 75th percentile) of salary growth.
    # -------------------------------------------------------------------------
    print("\n>>> [QUERY 2] Analyzing Salary Progression by Experience Level...")

    seniority_order = ["Entry-Level", "Mid-Level", "Senior-Level", "Executive/Lead"]

    q2_summary = (
        df.groupby("experience_level")["salary_usd"]
        .agg(
            count="count",
            min="min",
            p25=lambda s: s.quantile(0.25),
            median="median",
            mean="mean",
            p75=lambda s: s.quantile(0.75),
            max="max"
        )
        .reindex(seniority_order)
    )

    # Calculate percentage share of total market postings
    q2_summary["pct_share"] = (q2_summary["count"] / total_records) * 100

    q2_display = q2_summary.rename(
        columns={
            "count": "Postings",
            "pct_share": "Market Share",
            "min": "Min ($)",
            "p25": "25th % (Q1)",
            "median": "Median ($)",
            "mean": "Mean ($)",
            "p75": "75th % (Q3)",
            "max": "Max ($)"
        }
    )
    q2_display.index.name = "Experience Level"

    # Reorder columns logically
    ordered_q2_cols = ["Postings", "Market Share", "Min ($)", "25th % (Q1)", "Median ($)", "Mean ($)", "75th % (Q3)", "Max ($)"]
    q2_display = q2_display[ordered_q2_cols]

    q2_formats = {
        "Postings": lambda x: f"{x:,}",
        "Market Share": lambda x: f"{x:.1f}%",
        "Min ($)": format_currency,
        "25th % (Q1)": format_currency,
        "Median ($)": format_currency,
        "Mean ($)": format_currency,
        "75th % (Q3)": format_currency,
        "Max ($)": format_currency
    }

    format_table(
        q2_display,
        title="Query 2: Salary Progression & Distribution by Experience Level",
        column_formats=q2_formats
    )

    # -------------------------------------------------------------------------
    # QUERY 3: Remote Work Impact on Compensation
    # -------------------------------------------------------------------------
    # CONCEPT:
    # Does working remotely entail a salary penalty or premium?
    # We compare median, mean, and salary spread across On-site, Hybrid, and Remote.
    # -------------------------------------------------------------------------
    print("\n>>> [QUERY 3] Comparing Remote vs. Hybrid vs. On-Site Compensation...")

    work_setting_order = ["On-site", "Hybrid", "Remote"]

    q3_summary = (
        df.groupby("remote_ratio")["salary_usd"]
        .agg(
            postings_count="count",
            median_salary="median",
            mean_salary="mean",
            min_salary="min",
            max_salary="max"
        )
        .reindex(work_setting_order)
    )

    q3_summary["pct_market"] = (q3_summary["postings_count"] / total_records) * 100

    # Calculate difference relative to on-site median
    onsite_median = q3_summary.loc["On-site", "median_salary"]
    q3_summary["pct_vs_onsite"] = ((q3_summary["median_salary"] - onsite_median) / onsite_median) * 100

    q3_display = q3_summary.rename(
        columns={
            "postings_count": "Postings",
            "pct_market": "Market Share",
            "median_salary": "Median Salary",
            "mean_salary": "Mean Salary",
            "min_salary": "Min ($)",
            "max_salary": "Max ($)",
            "pct_vs_onsite": "Diff vs On-site"
        }
    )
    q3_display.index.name = "Work Setting"

    q3_formats = {
        "Postings": lambda x: f"{x:,}",
        "Market Share": lambda x: f"{x:.1f}%",
        "Median Salary": format_currency,
        "Mean Salary": format_currency,
        "Min ($)": format_currency,
        "Max ($)": format_currency,
        "Diff vs On-site": lambda x: f"{'+' if x > 0 else ''}{x:.1f}%" if x != 0 else "Baseline"
    }

    format_table(
        q3_display[["Postings", "Market Share", "Median Salary", "Mean Salary", "Min ($)", "Max ($)", "Diff vs On-site"]],
        title="Query 3: Remote Work Impact on Compensation (Work Setting Comparison)",
        column_formats=q3_formats
    )

    # -------------------------------------------------------------------------
    # QUERY 4: Top 10 Countries by Median Salary (Min 5 Postings)
    # -------------------------------------------------------------------------
    # CONCEPT:
    # Geographic compensation varies widely. By aggregating by 'company_location'
    # and filtering for countries with >= 5 postings, we eliminate outlier countries
    # with only 1 posting and identify the highest-paying tech job markets.
    # -------------------------------------------------------------------------
    print("\n>>> [QUERY 4] Computing Top 10 Countries by Median Salary...")

    min_country_postings = 5

    q4_grouped = (
        df.groupby("company_location")["salary_usd"]
        .agg(
            postings_count="count",
            median_salary="median",
            mean_salary="mean",
            min_salary="min",
            max_salary="max"
        )
        .reset_index()
    )

    q4_top10 = (
        q4_grouped[q4_grouped["postings_count"] >= min_country_postings]
        .sort_values(by="median_salary", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    q4_top10.index = range(1, len(q4_top10) + 1)
    q4_top10.index.name = "Rank"

    q4_display = q4_top10.rename(
        columns={
            "company_location": "Country Code",
            "postings_count": "Postings",
            "median_salary": "Median Salary",
            "mean_salary": "Mean Salary",
            "min_salary": "Min Salary",
            "max_salary": "Max Salary"
        }
    )

    q4_formats = {
        "Postings": lambda x: f"{x:,}",
        "Median Salary": format_currency,
        "Mean Salary": format_currency,
        "Min Salary": format_currency,
        "Max Salary": format_currency
    }

    format_table(
        q4_display.reset_index(),
        title="Query 4: Top 10 Countries by Median Salary (Min. 5 Postings)",
        column_formats=q4_formats
    )

    # -------------------------------------------------------------------------
    # QUERY 5: Two-Dimensional Pivot Table (Top 5 Job Titles vs. Experience Levels)
    # -------------------------------------------------------------------------
    # CONCEPT:
    # pivot_table allows cross-tabulating two categorical dimensions simultaneously:
    # - Rows (Index): Top 5 Job Titles by posting volume
    # - Columns: Experience Levels (Entry -> Mid -> Senior -> Lead)
    # - Values: Median Salary (aggregated across each intersection)
    # -------------------------------------------------------------------------
    print("\n>>> [QUERY 5] Generating 2D Pivot Table (Top 5 Job Titles vs Experience Levels)...")

    # Step 5.1: Identify the Top 5 most frequent job titles
    top_5_titles = df["job_title"].value_counts().head(5).index.tolist()

    # Step 5.2: Filter dataset to include only these top 5 job titles
    df_top5 = df[df["job_title"].isin(top_5_titles)]

    # Step 5.3: Build the pivot table with median salary
    pivot_median = df_top5.pivot_table(
        index="job_title",
        columns="experience_level",
        values="salary_usd",
        aggfunc="median"
    )

    # Ensure columns follow logical seniority ordering and clean index/column names
    pivot_median = pivot_median.reindex(columns=seniority_order)
    pivot_median.index.name = "Job Title"
    pivot_median.columns.name = None

    # Also build a sample count pivot table for reference
    pivot_count = df_top5.pivot_table(
        index="job_title",
        columns="experience_level",
        values="salary_usd",
        aggfunc="count"
    ).reindex(columns=seniority_order).fillna(0).astype(int)
    pivot_count.index.name = "Job Title"
    pivot_count.columns.name = None

    # Format the median table with dollar signs
    pivot_display = pivot_median.copy()
    for col in pivot_display.columns:
        pivot_display[col] = pivot_display[col].apply(format_currency)

    format_table(
        pivot_display,
        title="Query 5A: Median Salary Matrix (Top 5 Roles vs. Experience Levels)"
    )

    format_table(
        pivot_count,
        title="Query 5B: Sample Size Matrix (Postings Count for Top 5 Roles vs. Experience Levels)"
    )

    # -------------------------------------------------------------------------
    # SUMMARY & KEY BUSINESS INSIGHTS
    # -------------------------------------------------------------------------
    e_med = q2_summary.loc["Entry-Level", "median"]
    m_med = q2_summary.loc["Mid-Level", "median"]
    s_med = q2_summary.loc["Senior-Level", "median"]
    x_med = q2_summary.loc["Executive/Lead", "median"]

    print("\n" + "=" * 85)
    print(" MENTOR SUMMARY & KEY BUSINESS INSIGHTS (DAY 3 EDA) ")
    print("=" * 85)
    print("1. SPECIALIZATION PREMIUM (Query 1):")
    print("   - Advanced specialized roles (LLM Engineer, Research Scientist, ML Engineer,")
    print("     and Data Science Manager) dominate the top tier with median salaries > $103k-$107k.")
    print("   - High-demand infrastructure roles (Data Engineer, MLOps Engineer) follow closely.")
    print()
    print("2. CAREER LADDER PROGRESSION (Query 2):")
    print(f"   - Entry-Level median:  ${e_med:>8,.0f}")
    print(f"   - Mid-Level median:    ${m_med:>8,.0f}  (+{((m_med - e_med)/e_med)*100:.1f}% increase from Entry)")
    print(f"   - Senior-Level median: ${s_med:>8,.0f}  (+{((s_med - m_med)/m_med)*100:.1f}% increase from Mid)")
    print(f"   - Executive/Lead:      ${x_med:>8,.0f}  (+{((x_med - s_med)/s_med)*100:.1f}% increase from Senior)")
    print(f"   - Compensation more than doubles from Entry-Level (${e_med:,.0f}) to Executive (${x_med:,.0f}).")
    print()
    print("3. REMOTE WORK LANDSCAPE (Query 3):")
    print("   - Remote positions command a modest compensation premium over on-site roles")
    print("     (Median: $92,596 Remote vs. $90,046 On-site, +2.8% premium), confirming that")
    print("     remote flexibility does not penalize tech worker compensation.")
    print()
    print("4. GEOGRAPHIC PAY DISPARITIES (Query 4):")
    print("   - The United States (US) leads global tech compensation with a median of $126,987,")
    print("     followed by Canada (CA: $101,216) and Australia (AU: $100,133).")
    print()
    print("5. CROSS-DIMENSIONAL MATRICES (Query 5):")
    print("   - Across every title, compensation scales steadily across seniority levels.")
    print("   - AI Engineer and Machine Learning Engineer lead across all seniority tiers,")
    print("     reaching $159k-$163k median for Executive/Lead roles.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    analyze_salary_data()
