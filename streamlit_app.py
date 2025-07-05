"""
S&P 500 Stock Market Prediction - Streamlit Web Application
==========================================================

Interactive web application for exploring S&P 500 data and model predictions.
Perfect for data analyst portfolio demonstration.

Author: Data Analyst
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="S&P 500 Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load S&P 500 data with caching"""
    try:
        data = pd.read_csv("sp500.csv", index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.dropna()
        
        # Remove dividends and stock splits
        if 'Dividends' in data.columns:
            del data["Dividends"]
        if 'Stock Splits' in data.columns:
            del data["Stock Splits"]
            
        return data
    except:
        st.error("Error loading data. Please ensure sp500.csv is in the correct location.")
        return None

@st.cache_data
def prepare_features(data):
    """Prepare features for the model"""
    # Create target variable
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    data = data.loc["1990-01-01":].copy()
    
    # Feature engineering
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    
    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors.extend([ratio_column, trend_column])
    
    # Add volatility and momentum features
    data['Volatility_30'] = data['Close'].pct_change().rolling(30).std()
    data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
    new_predictors.extend(['Volatility_30', 'Momentum_20'])
    
    data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])
    
    return data, new_predictors

def train_model(data, predictors):
    """Train the Random Forest model"""
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(train[predictors], train["Target"])
    
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    
    return model, test, preds

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 S&P 500 Stock Market Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Data Analysis & Machine Learning Model")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Prepare features
    data, predictors = prepare_features(data)
    
    # Sidebar options
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Data Overview", "🤖 Model Performance", "📈 Interactive Charts", "📋 Detailed Report"]
    )
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data.index.min().date(), data.index.max().date()),
        min_value=data.index.min().date(),
        max_value=data.index.max().date()
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        filtered_data = data[(data.index.date >= date_range[0]) & (data.index.date <= date_range[1])]
    else:
        filtered_data = data
    
    # Main content based on selection
    if analysis_type == "📊 Data Overview":
        show_data_overview(filtered_data)
    
    elif analysis_type == "🤖 Model Performance":
        show_model_performance(data, predictors)
    
    elif analysis_type == "📈 Interactive Charts":
        show_interactive_charts(filtered_data)
    
    elif analysis_type == "📋 Detailed Report":
        show_detailed_report(data, predictors)

def show_data_overview(data):
    """Display data overview section"""
    st.header("📊 Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trading Days", f"{len(data):,}")
    
    with col2:
        st.metric("Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    with col3:
        st.metric("Average Close Price", f"${data['Close'].mean():.2f}")
    
    with col4:
        st.metric("Total Return", f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1):.1%}")
    
    # Price trend chart
    st.subheader("📈 S&P 500 Price Trend")
    fig = px.line(data, x=data.index, y='Close', title='S&P 500 Close Price Over Time')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.subheader("📋 Key Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Close Price': [
            f"${data['Close'].mean():.2f}",
            f"${data['Close'].median():.2f}",
            f"${data['Close'].std():.2f}",
            f"${data['Close'].min():.2f}",
            f"${data['Close'].max():.2f}"
        ],
        'Volume': [
            f"{data['Volume'].mean():,.0f}",
            f"{data['Volume'].median():,.0f}",
            f"{data['Volume'].std():,.0f}",
            f"{data['Volume'].min():,.0f}",
            f"{data['Volume'].max():,.0f}"
        ]
    })
    st.table(stats_df)

def show_model_performance(data, predictors):
    """Display model performance section"""
    st.header("🤖 Model Performance")
    
    # Train model
    with st.spinner("Training model..."):
        model, test, preds = train_model(data, predictors)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    precision = precision_score(test["Target"], preds)
    recall = recall_score(test["Target"], preds)
    f1 = f1_score(test["Target"], preds)
    accuracy = (test["Target"] == preds).mean()
    
    with col1:
        st.metric("Precision", f"{precision:.3f}")
    
    with col2:
        st.metric("Recall", f"{recall:.3f}")
    
    with col3:
        st.metric("F1-Score", f"{f1:.3f}")
    
    with col4:
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(test["Target"], preds)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Random Forest Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_interactive_charts(data):
    """Display interactive charts section"""
    st.header("📈 Interactive Charts")
    
    # Chart type selector
    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Price & Volume", "Returns Distribution", "Rolling Statistics", "Correlation Matrix"]
    )
    
    if chart_type == "Price & Volume":
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        fig.update_layout(title="S&P 500 Candlestick Chart", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig = px.bar(data, x=data.index, y='Volume', title="Trading Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Returns Distribution":
        # Daily returns
        daily_returns = data['Close'].pct_change().dropna()
        
        fig = px.histogram(
            daily_returns,
            nbins=50,
            title="Daily Returns Distribution",
            labels={'value': 'Daily Returns', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns over time
        fig = px.line(
            x=daily_returns.index,
            y=daily_returns.cumsum(),
            title="Cumulative Returns Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Rolling Statistics":
        # Rolling volatility
        rolling_vol = data['Close'].pct_change().rolling(30).std()
        fig = px.line(
            x=rolling_vol.index,
            y=rolling_vol,
            title="30-Day Rolling Volatility"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling mean
        rolling_mean = data['Close'].rolling(50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
        fig.add_trace(go.Scatter(x=data.index, y=rolling_mean, name='50-Day Moving Average'))
        fig.update_layout(title="Price with 50-Day Moving Average")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Correlation Matrix":
        # Calculate correlations
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_detailed_report(data, predictors):
    """Display detailed analysis report"""
    st.header("📋 Detailed Analysis Report")
    
    # Executive summary
    st.subheader("📊 Executive Summary")
    st.markdown("""
    This comprehensive analysis of the S&P 500 index demonstrates the application of machine learning 
    techniques to predict market direction. The model achieves competitive performance metrics while 
    providing actionable insights for trading strategies.
    """)
    
    # Methodology
    st.subheader("🔬 Methodology")
    st.markdown("""
    **Data Source**: Yahoo Finance S&P 500 historical data (1990-present)
    
    **Feature Engineering**:
    - Price ratios across multiple time horizons (2, 5, 60, 250, 1000 days)
    - Trend indicators based on historical market direction
    - Volatility measures (30-day rolling standard deviation)
    - Momentum indicators (5 and 20-day returns)
    
    **Model**: Random Forest Classifier with 200 trees
    **Validation**: Time-series split with 100-day test period
    **Threshold**: 60% probability for positive predictions
    """)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Market Behavior**:
        - Historical up/down ratio: {:.1%}
        - Average daily volatility: {:.2%}
        - Long-term growth trend: {:.1%} total return
        """.format(
            data['Target'].mean(),
            data['Close'].pct_change().std(),
            (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)
        ))
    
    with col2:
        st.markdown("""
        **Model Performance**:
        - Precision: Measures accuracy of positive predictions
        - Recall: Captures proportion of actual up days
        - F1-Score: Balanced measure of precision and recall
        - Feature importance reveals key market indicators
        """)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    st.markdown("""
    1. **Risk Management**: Use model predictions as one component of a comprehensive trading strategy
    2. **Portfolio Diversification**: Consider market timing based on model signals while maintaining diversification
    3. **Continuous Monitoring**: Regularly retrain the model with new data to maintain performance
    4. **Feature Analysis**: Monitor feature importance changes to adapt to evolving market conditions
    5. **Backtesting**: Always validate strategies on historical data before live implementation
    """)

if __name__ == "__main__":
    main() 