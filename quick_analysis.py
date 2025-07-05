"""
S&P 500 Stock Market Prediction - Quick Analysis
===============================================

Fast demonstration of data analysis skills for portfolio projects.
Optimized for quick execution while maintaining professional quality.

Author: Data Analyst
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def quick_analysis():
    """Perform quick but comprehensive analysis"""
    print("🚀 Starting Quick S&P 500 Analysis...")
    
    # Load data
    print("📊 Loading data...")
    data = pd.read_csv("sp500.csv", index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.dropna()
    
    # Remove unnecessary columns
    if 'Dividends' in data.columns:
        del data["Dividends"]
    if 'Stock Splits' in data.columns:
        del data["Stock Splits"]
    
    # Filter to recent data for faster processing
    data = data.loc["2010-01-01":].copy()
    
    # Create target variable
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    print(f"✅ Data loaded! Shape: {data.shape}")
    print(f"📅 Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    # Quick EDA
    print("📈 Creating visualizations...")
    create_quick_visualizations(data)
    
    # Feature engineering (simplified)
    print("⚙️ Engineering features...")
    predictors = create_simple_features(data)
    
    # Train model
    print("🤖 Training model...")
    model, predictions = train_simple_model(data, predictors)
    
    # Evaluate results
    print("📊 Evaluating results...")
    evaluate_results(predictions)
    
    # Generate insights
    print("💡 Generating insights...")
    generate_insights(data, model, predictors)
    
    print("🎉 Quick analysis completed!")

def create_quick_visualizations(data):
    """Create essential visualizations quickly"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Price trend
    axes[0, 0].plot(data.index, data['Close'], linewidth=1, color='blue')
    axes[0, 0].set_title('S&P 500 Close Price', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volume distribution
    axes[0, 1].hist(data['Volume'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Trading Volume Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Volume')
    axes[0, 1].set_ylabel('Frequency')
    
    # Daily returns
    daily_returns = data['Close'].pct_change().dropna()
    axes[1, 0].hist(daily_returns, bins=30, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Daily Returns Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Daily Returns')
    axes[1, 0].set_ylabel('Frequency')
    
    # Market direction
    market_direction = data['Target'].rolling(window=30).mean()
    axes[1, 1].plot(data.index, market_direction, linewidth=1, color='orange')
    axes[1, 1].set_title('30-Day Market Direction Trend', fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Up Days Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_simple_features(data):
    """Create simplified features for faster processing"""
    predictors = []
    
    # Simple price ratios
    for horizon in [5, 20, 60]:
        rolling_avg = data.rolling(horizon).mean()
        ratio_col = f"Price_Ratio_{horizon}"
        data[ratio_col] = data["Close"] / rolling_avg["Close"]
        predictors.append(ratio_col)
        
        trend_col = f"Trend_{horizon}"
        data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
        predictors.append(trend_col)
    
    # Volatility
    data['Volatility'] = data['Close'].pct_change().rolling(20).std()
    predictors.append('Volatility')
    
    # Momentum
    data['Momentum'] = data['Close'] / data['Close'].shift(10) - 1
    predictors.append('Momentum')
    
    # Remove NaN values
    data = data.dropna(subset=predictors + ['Target'])
    
    print(f"✅ Created {len(predictors)} features")
    return predictors

def train_simple_model(data, predictors):
    """Train a simplified model"""
    # Use smaller test set for faster processing
    train = data.iloc[:-50]
    test = data.iloc[-50:]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train[predictors], train["Target"])
    
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    
    predictions = pd.DataFrame({
        'Target': test["Target"],
        'Predictions': preds
    }, index=test.index)
    
    return model, predictions

def evaluate_results(predictions):
    """Evaluate model performance"""
    precision = precision_score(predictions['Target'], predictions['Predictions'])
    recall = recall_score(predictions['Target'], predictions['Predictions'])
    f1 = f1_score(predictions['Target'], predictions['Predictions'])
    accuracy = (predictions['Target'] == predictions['Predictions']).mean()
    
    print(f"\n🎯 Model Performance:")
    print(f"   • Precision: {precision:.3f}")
    print(f"   • Recall: {recall:.3f}")
    print(f"   • F1-Score: {f1:.3f}")
    print(f"   • Accuracy: {accuracy:.3f}")
    
    # Quick confusion matrix
    cm = confusion_matrix(predictions['Target'], predictions['Predictions'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('quick_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_insights(data, model, predictors):
    """Generate key insights"""
    print(f"\n💡 Key Insights:")
    print(f"   • Total trading days analyzed: {len(data):,}")
    print(f"   • Market up probability: {data['Target'].mean():.1%}")
    print(f"   • Average daily volume: {data['Volume'].mean():,.0f}")
    print(f"   • Total return: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1):.1%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔝 Top 5 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
        print(f"   {i}. {row['Feature']}: {row['Importance']:.3f}")
    
    # Train model again to get predictions for insights
    train = data.iloc[:-50]
    test = data.iloc[-50:]
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    predictions = pd.DataFrame({
        'Target': test["Target"],
        'Predictions': preds
    }, index=test.index)
    
    # Save insights
    insights = f"""
# Quick S&P 500 Analysis Insights

## Model Performance
- Precision: {precision_score(predictions['Target'], predictions['Predictions']):.3f}
- Recall: {recall_score(predictions['Target'], predictions['Predictions']):.3f}
- F1-Score: {f1_score(predictions['Target'], predictions['Predictions']):.3f}

## Key Findings
- Market up probability: {data['Target'].mean():.1%}
- Total return since 2010: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1):.1%}
- Most important feature: {feature_importance.iloc[0]['Feature']}

## Business Applications
1. Market timing strategies
2. Risk management
3. Portfolio optimization
4. Trading automation foundation

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    with open('quick_insights.md', 'w') as f:
        f.write(insights)
    
    print(f"\n✅ Quick insights saved to 'quick_insights.md'")

if __name__ == "__main__":
    quick_analysis() 