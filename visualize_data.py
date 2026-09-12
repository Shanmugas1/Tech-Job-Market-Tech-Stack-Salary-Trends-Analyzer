"""
Tech Job Market & Salary Trends Analyzer
Script: visualize_data.py
Description: Day 4 - Interactive Data Visualization with Plotly Express
Dependencies: pandas, plotly
"""

import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set a sleek, modern default template for all Plotly visuals
pio.templates.default = "plotly_white"


# =============================================================================
# MENTOR LESSON: CONCEPTUAL FOUNDATIONS OF INTERACTIVE DATA VISUALIZATION
# =============================================================================
# 1. WHY PLOTLY EXPRESS (px) OVER STATIC PLOTTING LIBRARIES?
#    - Declarative & High-Level: Create complex, publication-ready interactive
#      visualizations in 1-2 lines of clean code.
#    - Interactive by Default: Native zooming, panning, box selection, dynamic
#      tooltips (hover templates), and isolate/toggle legends without extra code.
#    - Production-Ready: Plotly figures translate directly into web applications,
#      Dashboards (Streamlit, Dash), and shareable HTML reports.
#
# 2. WHAT DOES A BOX PLOT (px.box) REPRESENT VS. A STANDARD BAR CHART?
#    - A Bar Chart only shows a SINGLE summary statistic (e.g., the mean or median),
#      completely hiding the underlying distribution, spread, and variance.
#    - A Box Plot (Tukey 5-Number Summary) visualizes the COMPLETE distribution:
#      * Lower Whisker: Minimum value (excluding statistical outliers)
#      * Box Bottom: 25th Percentile (Q1 - first quartile)
#      * Middle Line: 50th Percentile (Median - robust central tendency)
#      * Box Top: 75th Percentile (Q3 - third quartile)
#      * Interquartile Range (IQR): The middle 50% of the compensation spread (Q3 - Q1)
#      * Upper Whisker: Maximum value within 1.5 * IQR
#      * Outlier Points: Individual dots representing extreme compensation packages.
#
# 3. HOW DOES SAVING CHARTS AS STANDALONE HTML WORK?
#    - Plotly embeds the chart specification (JSON) alongside the Plotly.js engine
#      inside a single self-contained HTML file.
#    - BENEFIT: Anyone can double-click and view fully interactive charts in any
#      modern browser (Chrome, Edge, Safari, Firefox) with ZERO Python/Jupyter installation.
# =============================================================================


def generate_interactive_visualizations():
    """
    Loads cleaned salary data, generates 4 interactive Plotly Express charts,
    and exports them as standalone HTML files inside the outputs/ directory.
    """
    print("\n" + "=" * 80)
    print(" DAY 4: INTERACTIVE DATA VISUALIZATION PIPELINE (PLOTLY EXPRESS) ")
    print(" Project: Tech Job Market & Salary Trends Analyzer")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 0: Verify Paths & Create Output Directory
    # -------------------------------------------------------------------------
    data_path = os.path.join("data", "cleaned_salaries.csv")
    output_dir = "outputs"

    if not os.path.exists(data_path):
        print(f"\n[ERROR] Cleaned dataset not found at: '{data_path}'")
        print("Please run 'python clean_data.py' first to generate the cleaned dataset.\n")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[STEP 0] Output directory verified: '{output_dir}/'")

    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"[STEP 0] Successfully loaded {total_records:,} cleaned records from '{data_path}'.\n")

    # Consistent seniority ordering for domain accuracy
    seniority_order = ["Entry-Level", "Mid-Level", "Senior-Level", "Executive/Lead"]

    # -------------------------------------------------------------------------
    # CHART 1: Top 10 Highest-Paying Job Titles (Horizontal Bar Chart)
    # -------------------------------------------------------------------------
    print(">>> [CHART 1/4] Generating Top 10 Highest-Paying Job Titles Chart...")

    min_role_postings = 5
    role_agg = (
        df.groupby("job_title")["salary_usd"]
        .agg(postings="count", median_salary="median", mean_salary="mean")
        .reset_index()
    )

    # Filter roles with >= 5 postings to eliminate single-sample anomalies
    top10_roles = (
        role_agg[role_agg["postings"] >= min_role_postings]
        .sort_values(by="median_salary", ascending=True)  # ascending=True puts highest at the top in horizontal bar
        .tail(10)
        .reset_index(drop=True)
    )

    fig1 = px.bar(
        top10_roles,
        x="median_salary",
        y="job_title",
        orientation="h",
        text="median_salary",
        color="median_salary",
        color_continuous_scale="Blues",
        labels={"median_salary": "Median Salary (USD)", "job_title": "Job Title", "postings": "Postings"},
        title="<b>Top 10 Highest-Paying Tech Job Titles (2026)</b><br><sup>Filtered for roles with at least 5 job postings</sup>",
        custom_data=["postings", "mean_salary"]
    )

    # Format data labels on bars and hover card tooltip
    fig1.update_traces(
        texttemplate="$%{x:,.0f}",
        textposition="inside",
        textfont_color="white",
        hovertemplate=(
            "<b>%{y}</b><br><br>"
            "Median Salary: <b>$%{x:,.0f}</b><br>"
            "Mean Salary: <b>$%{customdata[1]:,.0f}</b><br>"
            "Sample Size: <b>%{customdata[0]:,} postings</b>"
            "<extra></extra>"
        )
    )

    fig1.update_layout(
        xaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Annual Salary (USD)"),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
        height=520,
        margin=dict(l=180, r=40, t=80, b=50)
    )

    chart1_path = os.path.join(output_dir, "top_paying_roles.html")
    fig1.write_html(chart1_path)
    print(f" -> [SAVED] Chart 1 saved to: '{chart1_path}' ({os.path.getsize(chart1_path):,} bytes)")

    # -------------------------------------------------------------------------
    # CHART 2: Salary Distribution Across Experience Levels (Interactive Box Plot)
    # -------------------------------------------------------------------------
    print("\n>>> [CHART 2/4] Generating Salary Distribution by Experience Level Box Plot...")

    # Color palette tailored for career progression
    exp_color_map = {
        "Entry-Level": "#60A5FA",       # Light Blue
        "Mid-Level": "#3B82F6",         # Primary Blue
        "Senior-Level": "#1D4ED8",      # Deep Navy
        "Executive/Lead": "#4338CA"     # Indigo
    }

    fig2 = px.box(
        df,
        x="experience_level",
        y="salary_usd",
        color="experience_level",
        category_orders={"experience_level": seniority_order},
        color_discrete_map=exp_color_map,
        points="outliers",  # Clearly display outlier compensation packages
        labels={"experience_level": "Experience Level", "salary_usd": "Annual Salary (USD)"},
        title="<b>Salary Distribution & IQR Spread Across Experience Levels</b><br><sup>Visualizing Median, Interquartile Range (Q1-Q3), and Outliers</sup>"
    )

    fig2.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Salary: <b>$%{y:,.0f}</b>"
            "<extra></extra>"
        )
    )

    fig2.update_layout(
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Annual Salary (USD)"),
        xaxis=dict(title="Career Seniority Tier"),
        showlegend=False,
        height=540,
        margin=dict(l=60, r=40, t=80, b=50)
    )

    chart2_path = os.path.join(output_dir, "salary_by_experience.html")
    fig2.write_html(chart2_path)
    print(f" -> [SAVED] Chart 2 saved to: '{chart2_path}' ({os.path.getsize(chart2_path):,} bytes)")

    # -------------------------------------------------------------------------
    # CHART 3: Remote Work Impact on Compensation (Bar Chart)
    # -------------------------------------------------------------------------
    print("\n>>> [CHART 3/4] Generating Remote Work Compensation Impact Chart...")

    work_setting_order = ["On-site", "Hybrid", "Remote"]
    remote_agg = (
        df.groupby("remote_ratio")["salary_usd"]
        .agg(postings="count", median_salary="median", mean_salary="mean")
        .reindex(work_setting_order)
        .reset_index()
    )

    remote_agg["pct_share"] = (remote_agg["postings"] / total_records) * 100

    remote_colors = {
        "On-site": "#0D9488",   # Teal
        "Hybrid": "#06B6D4",    # Cyan
        "Remote": "#3B82F6"     # Blue
    }

    fig3 = px.bar(
        remote_agg,
        x="remote_ratio",
        y="median_salary",
        color="remote_ratio",
        color_discrete_map=remote_colors,
        text="median_salary",
        labels={"remote_ratio": "Work Setting", "median_salary": "Median Salary (USD)"},
        title="<b>Remote Work Impact on Tech Compensation</b><br><sup>Comparing Median Salary across On-site, Hybrid, and Remote Work Models</sup>",
        custom_data=["postings", "pct_share", "mean_salary"]
    )

    fig3.update_traces(
        texttemplate="$%{y:,.0f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x} Setting</b><br><br>"
            "Median Salary: <b>$%{y:,.0f}</b><br>"
            "Mean Salary: <b>$%{customdata[2]:,.0f}</b><br>"
            "Market Volume: <b>%{customdata[0]:,} postings (%{customdata[1]:.1f}%)</b>"
            "<extra></extra>"
        )
    )

    fig3.update_layout(
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Annual Salary (USD)", range=[0, remote_agg["median_salary"].max() * 1.18]),
        xaxis=dict(title="Workplace Arrangement"),
        showlegend=False,
        height=500,
        margin=dict(l=60, r=40, t=80, b=50)
    )

    chart3_path = os.path.join(output_dir, "salary_by_remote_work.html")
    fig3.write_html(chart3_path)
    print(f" -> [SAVED] Chart 3 saved to: '{chart3_path}' ({os.path.getsize(chart3_path):,} bytes)")

    # -------------------------------------------------------------------------
    # CHART 4: Top 10 Countries by Median Salary (Bar Chart)
    # -------------------------------------------------------------------------
    print("\n>>> [CHART 4/4] Generating Top 10 Countries by Median Salary Chart...")

    min_country_postings = 5
    country_agg = (
        df.groupby("company_location")["salary_usd"]
        .agg(postings="count", median_salary="median", mean_salary="mean")
        .reset_index()
    )

    top10_countries = (
        country_agg[country_agg["postings"] >= min_country_postings]
        .sort_values(by="median_salary", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    fig4 = px.bar(
        top10_countries,
        x="company_location",
        y="median_salary",
        color="median_salary",
        color_continuous_scale="Viridis",
        text="median_salary",
        labels={"company_location": "Country Code", "median_salary": "Median Salary (USD)"},
        title="<b>Top 10 Countries by Median Tech Salary (2026)</b><br><sup>Filtered for markets with at least 5 job postings in dataset</sup>",
        custom_data=["postings", "mean_salary"]
    )

    fig4.update_traces(
        texttemplate="$%{y:,.0f}",
        textposition="outside",
        hovertemplate=(
            "<b>Country: %{x}</b><br><br>"
            "Median Salary: <b>$%{y:,.0f}</b><br>"
            "Mean Salary: <b>$%{customdata[1]:,.0f}</b><br>"
            "Postings: <b>%{customdata[0]:,}</b>"
            "<extra></extra>"
        )
    )

    fig4.update_layout(
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Annual Salary (USD)", range=[0, top10_countries["median_salary"].max() * 1.18]),
        xaxis=dict(title="Company Country Code"),
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=60, r=40, t=80, b=50)
    )

    chart4_path = os.path.join(output_dir, "top_paying_countries.html")
    fig4.write_html(chart4_path)
    print(f" -> [SAVED] Chart 4 saved to: '{chart4_path}' ({os.path.getsize(chart4_path):,} bytes)")

    # -------------------------------------------------------------------------
    # SUMMARY & COMPLETION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ALL 4 INTERACTIVE HTML VISUALIZATIONS GENERATED SUCCESSFULLY! ")
    print("=" * 80)
    print("Generated Artifacts in 'outputs/':")
    print(f"  1. Top Paying Job Titles:     {os.path.abspath(chart1_path)}")
    print(f"  2. Salary by Experience (Box): {os.path.abspath(chart2_path)}")
    print(f"  3. Remote Work Comparison:    {os.path.abspath(chart3_path)}")
    print(f"  4. Top Paying Countries:      {os.path.abspath(chart4_path)}")
    print("\nTip: Double-click any of the .html files above to open them in your web browser!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    generate_interactive_visualizations()
