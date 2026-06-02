# Step 1: Setup and Data Preparation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import agent.retrieval as retrieval
import agent.agent as agent_runner

# Set page config
st.set_page_config(page_title="Store Sales Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Load your preprocessed data
    # Adjust paths to where your data files are located
    train_merged = pd.read_csv('store_sales/trainMerged.csv', parse_dates=['date'])
    holidays_processed = pd.read_csv('store_sales/holidaysAggregate.csv', parse_dates=['date'])
    oil = pd.read_csv('store_sales/oil.csv', parse_dates=['date'])
    stores = pd.read_csv('store_sales/stores.csv')
    holidays = pd.read_csv('store_sales/holidays_events.csv', parse_dates=['date'])
    forecast = pd.read_csv('store_sales/predictions_xgboost.csv', parse_dates=['date'])  # if you have a forecast file
    
    # Ensure date columns are datetime
    train_merged['date'] = pd.to_datetime(train_merged['date'])
    
    # Create additional features (payday, etc.) if not already present
    if 'is_payday' not in train_merged.columns:
        train_merged['day'] = train_merged['date'].dt.day
        train_merged['days_in_month'] = train_merged['date'].dt.daysinmonth
        train_merged['is_last_day'] = train_merged['day'] == train_merged['days_in_month']
        train_merged['is_payday'] = (train_merged['day'] == 15) | train_merged['is_last_day']

    retrieval.setup_db(train_merged, stores, oil, holidays,forecast)

    return train_merged, holidays_processed, oil, stores

train_merged, holidays_processed, oil, stores = load_data()

@st.cache_data
def load_predictions():
    pred_df = pd.read_csv('store_sales/predictions_xgboost.csv', parse_dates=['date'])
    return pred_df

# Add this after loading the other data
predictions = load_predictions()


# Step 2: Sidebar Filters

st.sidebar.header("Filters")

# Get unique values
cities = train_merged['city'].unique()
families = train_merged['family'].unique()
min_date = train_merged['date'].min().date()
max_date = train_merged['date'].max().date()

# Multi-select for cities and families
selected_cities = st.sidebar.multiselect("Select Cities", cities, default=cities[:3])
selected_families = st.sidebar.multiselect("Select Product Families", families, default=families[:5])

# Date range
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Toggle to include only paydays or holidays
show_payday_only = st.sidebar.checkbox("Show only paydays")
show_holiday_only = st.sidebar.checkbox("Show only holidays")

# Apply filters
filtered_data = train_merged[
    (train_merged['city'].isin(selected_cities)) &
    (train_merged['family'].isin(selected_families)) &
    (train_merged['date'].between(pd.Timestamp(start_date), pd.Timestamp(end_date)))
]

if show_payday_only:
    filtered_data = filtered_data[filtered_data['is_payday']]
if show_holiday_only:
    filtered_data = filtered_data[filtered_data['is_holiday']]

# Step 3: Main KPIs Row

st.title("🏪 NitiAI")
st.markdown("---")

# Compute KPIs on filtered data
total_sales = filtered_data['sales'].sum()
avg_daily = filtered_data.groupby('date')['sales'].sum().mean() if not filtered_data.empty else 0
num_stores = filtered_data['store_nbr'].nunique()
num_families = filtered_data['family'].nunique()
num_days = filtered_data['date'].nunique()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Avg Daily Sales", f"${avg_daily:,.0f}")
col3.metric("Stores", num_stores)
col4.metric("Families", num_families)
col5.metric("Days", num_days)

# Step 4: Tabs for Detailed Analysis

tab1, tab2, tab3, tab4, tab5, tab6, tab_ask = st.tabs([
    "📅 Sales Over Time", 
    "🎉 Holiday Impact", 
    "💰 Payday Impact", 
    "🏙️ City & Family", 
    "📦 Top Performers",
    "🔮 Sales Forecast",
    "Ask NitiAI"
])

# Tab 1: Sales Over Time

with tab1:
    st.subheader("Daily Sales Trend")
    
    # Aggregate daily sales
    daily_sales = filtered_data.groupby('date')['sales'].sum().reset_index()
    
    fig = px.line(daily_sales, x='date', y='sales', title='Daily Sales')
    st.plotly_chart(fig, use_container_width=True)
    
    # Option to view by city
    if st.checkbox("Show by city"):
        city_daily = filtered_data.groupby(['date', 'city'])['sales'].sum().reset_index()
        fig_city = px.line(city_daily, x='date', y='sales', color='city', title='Daily Sales by City')
        st.plotly_chart(fig_city, use_container_width=True)

# Tab 2: Holiday Impact

with tab2:
    st.subheader("Holiday Impact Analysis")
    
    # Compute holiday vs non-holiday averages
    holiday_stats = filtered_data.groupby('is_holiday')['sales'].agg(['mean', 'count']).reset_index()
    holiday_stats['is_holiday'] = holiday_stats['is_holiday'].map({True: 'Holiday', False: 'Non-Holiday'})
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(holiday_stats, x='is_holiday', y='mean', 
                     title='Average Sales per Transaction',
                     labels={'mean': 'Avg Sales ($)', 'is_holiday': ''})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top holidays by sales
        top_holidays = filtered_data[filtered_data['is_holiday']].groupby('holiday_description')['sales'].mean().nlargest(10).reset_index()
        fig2 = px.bar(top_holidays, x='sales', y='holiday_description', orientation='h',
                     title='Top 10 Holidays by Avg Sales')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Transferred vs non-transferred
    # if st.checkbox("Compare transferred holidays"):
    #     trans_stats = filtered_data[filtered_data['is_holiday']].groupby('was_transferred')['sales'].mean().reset_index()
    #     trans_stats['was_transferred'] = trans_stats['was_transferred'].map({True: 'Transferred', False: 'Non-Transferred'})
    #     fig3 = px.bar(trans_stats, x='was_transferred', y='sales', title='Transferred vs Non-Transferred Holidays')
    #     st.plotly_chart(fig3, use_container_width=True)
    # with tab2:
    # # ... existing code ...

    st.subheader("Holiday Sales Lift by City")
    # Compute lift by city
    holiday_lift_city = (
        filtered_data.groupby(['city', 'is_holiday'])['sales']
        .mean()
        .unstack()
        .rename(columns={True: 'holiday_avg', False: 'non_holiday_avg'})
    )
    holiday_lift_city['lift_%'] = (
        (holiday_lift_city['holiday_avg'] / holiday_lift_city['non_holiday_avg'] - 1) * 100
    ).round(1)
    holiday_lift_city = holiday_lift_city.dropna(subset=['lift_%']).sort_values('lift_%', ascending=False)

    if not holiday_lift_city.empty:
        fig_lift = px.bar(
            holiday_lift_city.reset_index(),
            x='city',
            y='lift_%',
            title='Holiday Sales Lift (%) by City',
            labels={'lift_%': 'Lift (%)', 'city': ''},
            color='lift_%',
            color_continuous_scale='RdYlGn'
        )
        fig_lift.update_layout(showlegend=False)
        st.plotly_chart(fig_lift, use_container_width=True)
    else:
        st.info("No holiday data for selected filters.")

# Tab 3: Payday Impact

with tab3:
    st.subheader("Payday Impact Analysis")
    
    payday_stats = filtered_data.groupby('is_payday')['sales'].agg(['mean', 'count']).reset_index()
    payday_stats['is_payday'] = payday_stats['is_payday'].map({True: 'Payday', False: 'Non-Payday'})
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(payday_stats, x='is_payday', y='mean',
                     title='Average Sales per Transaction',
                     labels={'mean': 'Avg Sales ($)', 'is_payday': ''})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payday lift over time
        monthly = filtered_data.groupby([filtered_data['date'].dt.to_period('M'), 'is_payday'])['sales'].mean().unstack().fillna(0)
        monthly.index = monthly.index.astype(str)
        fig2 = px.line(monthly, x=monthly.index, y=[True, False], 
                      labels={'value': 'Avg Sales', 'variable': 'Payday'})
        fig2.update_layout(xaxis_title='Month')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Payday Impact Lift by City")
    payday_lift_city = (
        filtered_data.groupby(['city', 'is_payday'])['sales']
        .mean()
        .unstack()
        .rename(columns={True: 'payday_avg', False: 'non_payday_avg'})
    )
    payday_lift_city['lift_%'] = (
        (payday_lift_city['payday_avg'] / payday_lift_city['non_payday_avg'] - 1) * 100
    ).round(1)
    payday_lift_city = payday_lift_city.dropna(subset=['lift_%']).sort_values('lift_%', ascending=False)

    if not payday_lift_city.empty:
        fig_lift = px.bar(
            payday_lift_city.reset_index(),
            x='city',
            y='lift_%',
            title='Payday Sales Lift (%) by City',
            labels={'lift_%': 'Lift (%)', 'city': ''},
            color='lift_%',
            color_continuous_scale='RdYlGn'
        )
        fig_lift.update_layout(showlegend=False)
        st.plotly_chart(fig_lift, use_container_width=True)
    else:
        st.info("No payday data for selected filters.")

# Tab 4: City & Family Deep Dive
with tab4:
    st.subheader("City and Family Performance")
    
    # Heatmap of average sales by city and family
    pivot = filtered_data.pivot_table(values='sales', index='family', columns='city', aggfunc='mean').fillna(0)
    fig = px.imshow(pivot, text_auto='.0f', aspect="auto",
                   title='Average Sales per Transaction (City vs Family)',
                   labels=dict(x="City", y="Family", color="Avg Sales"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series for selected city-family
    st.subheader("Time Series for Specific City-Family")
    sel_city = st.selectbox("Select City", selected_cities)
    sel_family = st.selectbox("Select Family", selected_families)
    
    series = filtered_data[(filtered_data['city'] == sel_city) & (filtered_data['family'] == sel_family)]
    series_daily = series.groupby('date')['sales'].sum().reset_index()
    if not series_daily.empty:
        fig2 = px.line(series_daily, x='date', y='sales', title=f'{sel_family} in {sel_city}')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No data for selected combination.")

# Tab 5: Top Performers

with tab5:
    st.subheader("Top Cities and Families by Total Sales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city_totals = filtered_data.groupby('city')['sales'].sum().nlargest(10).reset_index()
        fig = px.bar(city_totals, x='sales', y='city', orientation='h', title='Top Cities')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        family_totals = filtered_data.groupby('family')['sales'].sum().nlargest(10).reset_index()
        fig2 = px.bar(family_totals, x='sales', y='family', orientation='h', title='Top Families')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Top city for each family
    st.subheader("Top City for Each Family (by total sales)")
    top_city_per_family = filtered_data.groupby(['family', 'city'])['sales'].sum().reset_index()
    top_city_per_family = top_city_per_family.loc[
        top_city_per_family.groupby('family')['sales'].idxmax()
    ].sort_values('sales', ascending=False)
    top_city_per_family['sales'] = top_city_per_family['sales'].apply('${:,.0f}'.format)
    st.dataframe(top_city_per_family[['family', 'city', 'sales']], use_container_width=True)

with tab6:
    st.subheader("Sales Forecast (Aug 16–31, 2017)")
    
    # Optional: allow user to select forecast model if you have multiple
    # model_options = ["XGBoost"]  # extend with Prophet, SARIMA, etc.
    # selected_model = st.selectbox("Select forecast model", model_options)
    
    # Load the corresponding predictions
    # if selected_model == "XGBoost":
    forecast_df = predictions.copy()
    # elif selected_model == "Prophet":
    #     forecast_df = pd.read_csv('prophet_predictions.csv', parse_dates=['date'])
    
    # Date range filter for forecast period
    # min_fc_date = forecast_df['date'].min().date()
    # max_fc_date = forecast_df['date'].max().date()
    # fc_start, fc_end = st.date_input(
    #     "Forecast date range",
    #     value=[min_fc_date, max_fc_date],
    #     min_value=min_fc_date,
    #     max_value=max_fc_date
    # )
    # mask = (forecast_df['date'] >= pd.Timestamp(fc_start)) & (forecast_df['date'] <= pd.Timestamp(fc_end))
    # filtered_forecast = forecast_df[mask]
    
    # Plot forecast
    fig_forecast = px.line(
        forecast_df,
        x='date',
        y='predicted_total_sales',
        title='Predicted Total Daily Sales',
        labels={'predicted_total_sales': 'Predicted Sales ($)', 'date': 'Date'},
        markers=True
    )
    fig_forecast.update_traces(line=dict(color='#2E86AB', width=2), marker=dict(size=6))
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Show actual historical sales for comparison (if available)
    # If you have actual sales for the same dates (e.g., from validation set), load them.
    # For demonstration, we assume you have a DataFrame `actual_validation` with columns 'date' and 'sales'
    # Uncomment the following block if you have actual data:
    # """
    # actual_validation = pd.read_csv('validation_actual.csv', parse_dates=['date'])
    # actual_validation = actual_validation[(actual_validation['date'] >= fc_start) & (actual_validation['date'] <= fc_end)]
    # if not actual_validation.empty:
    #     fig_compare = px.line(
    #         actual_validation, x='date', y='sales',
    #         title='Actual vs Predicted',
    #         labels={'sales': 'Actual Sales ($)'}
    #     )
    #     fig_compare.add_scatter(x=filtered_forecast['date'], y=filtered_forecast['predicted_total_sales'],
    #                             mode='lines+markers', name='Predicted')
    #     st.plotly_chart(fig_compare, use_container_width=True)
    # else:
    #     st.info("No actual sales data available for the forecast period.")
    # """
    
    # Display forecast as a table
    st.subheader("Forecast Table")
    st.dataframe(
        forecast_df.rename(columns={'predicted_total_sales': 'Predicted Sales ($)'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download forecast as CSV",
        csv,
        "xgboost_forecast.csv",
        "text/csv"
    )

# ── Ask NitiAI tab ────────────────────────────────────────────────────────
with tab_ask:  # add "Ask NitiAI" to your existing st.tabs([...]) call
    st.markdown("### Ask NitiAI")
    st.caption("Ask any question about your sales data in plain English.")

    # Example questions for easy onboarding
    examples = [
        "Why did sales drop in Quito last month?",
        "Compare Quito vs Guayaquil in July 2017",
        "Which stores underperformed in Quito in August 2017?",
        "How did the oil price affect sales in 2017?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["niti_question"] = ex

    question = st.text_input(
        "Your question",
        value=st.session_state.get("niti_question", ""),
        placeholder="e.g. Why did sales drop in Quito last month?",
        key="niti_input",
    )

    if st.button("Ask NitiAI", type="primary") and question:
        with st.spinner("Analysing..."):
            answer, findings = agent_runner.run(question)

        # ── Response ───────────────────────────────────────────────────────
        st.markdown(answer)

        # # ── Supporting chart: actual vs forecast ───────────────────────────
        if findings.get("forecast_data"):
            import plotly.graph_objects as go
            import pandas as pd

            fva = pd.DataFrame(findings["forecast_data"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fva["date"], y=fva["actual"],
                name="Actual", line=dict(color="#1D9E75", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=fva["date"], y=fva["forecast"],
                name="Forecast", line=dict(color="#7F77DD", width=2, dash="dot")
            ))
            fig.update_layout(
                title=f"Actual vs forecast — {findings['city']}",
                xaxis_title="Date", yaxis_title="Sales ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=320, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Underperforming stores table ───────────────────────────────────
        if findings.get("underperforming_stores"):
            import pandas as pd
            st.markdown("**Stores below city average**")
            st.dataframe(
                pd.DataFrame(findings["underperforming_stores"]),
                use_container_width=True,
                hide_index=True,
            )