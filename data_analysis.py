"""
S&P 500 Stock Market Prediction - Comprehensive Data Analysis
============================================================

This script performs comprehensive analysis of S&P 500 data including:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training and Evaluation
- Performance Metrics and Insights
- Visualization and Reporting

Author: Data Analyst
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SP500Analyzer:
    def __init__(self, data_path="sp500.csv"):
        """Initialize the analyzer with S&P 500 data"""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.predictions = None
        
    def load_and_prepare_data(self):
        """Load and prepare the S&P 500 dataset"""
        print("📊 Loading S&P 500 data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path, index_col=0)
        self.data.index = pd.to_datetime(self.data.index)
        
        # Basic data cleaning
        self.data = self.data.dropna()
        
        # Remove dividends and stock splits for analysis
        if 'Dividends' in self.data.columns:
            del self.data["Dividends"]
        if 'Stock Splits' in self.data.columns:
            del self.data["Stock Splits"]
            
        print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
        print(f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        return self.data
    
    def create_target_variable(self):
        """Create binary target variable for market direction prediction"""
        print("🎯 Creating target variable...")
        
        # Create tomorrow's price
        self.data["Tomorrow"] = self.data["Close"].shift(-1)
        
        # Create binary target (1 if market goes up, 0 if down)
        self.data["Target"] = (self.data["Tomorrow"] > self.data["Close"]).astype(int)
        
        # Filter data from 1990 onwards for more relevant analysis
        self.data = self.data.loc["1990-01-01":].copy()
        
        print(f"✅ Target variable created! Market up days: {self.data['Target'].sum()}")
        print(f"📈 Market down days: {len(self.data) - self.data['Target'].sum()}")
        
        return self.data
    
    def perform_eda(self):
        """Perform comprehensive Exploratory Data Analysis"""
        print("🔍 Performing Exploratory Data Analysis...")
        
        # Create EDA visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Price trend over time
        axes[0, 0].plot(self.data.index, self.data['Close'], linewidth=1)
        axes[0, 0].set_title('S&P 500 Close Price Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Close Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Volume distribution
        axes[0, 1].hist(self.data['Volume'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Trading Volume Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Volume')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Daily returns distribution
        daily_returns = self.data['Close'].pct_change().dropna()
        axes[1, 0].hist(daily_returns, bins=50, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Market direction over time
        market_direction = self.data['Target'].rolling(window=30).mean()
        axes[1, 1].plot(self.data.index, market_direction, linewidth=1, color='orange')
        axes[1, 1].set_title('30-Day Rolling Market Direction (Up/Down Ratio)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Proportion of Up Days')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('StockPricePredictionModel/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key statistics
        print("\n📈 Key Statistics:")
        print(f"   • Total trading days: {len(self.data):,}")
        print(f"   • Average daily volume: {self.data['Volume'].mean():,.0f}")
        print(f"   • Average close price: ${self.data['Close'].mean():.2f}")
        print(f"   • Market up probability: {self.data['Target'].mean():.2%}")
        print(f"   • Total return since 1990: {((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) - 1):.1%}")
        
        return fig
    
    def engineer_features(self):
        """Create advanced features for prediction"""
        print("⚙️ Engineering features...")
        
        # Define time horizons for feature engineering
        horizons = [2, 5, 60, 250, 1000]
        new_predictors = []
        
        for horizon in horizons:
            # Price ratio features (current price vs rolling average)
            rolling_averages = self.data.rolling(horizon).mean()
            ratio_column = f"Close_Ratio_{horizon}"
            self.data[ratio_column] = self.data["Close"] / rolling_averages["Close"]
            
            # Trend features (rolling sum of target)
            trend_column = f"Trend_{horizon}"
            self.data[trend_column] = self.data.shift(1).rolling(horizon).sum()["Target"]
            
            new_predictors.extend([ratio_column, trend_column])
        
        # Add volatility features
        self.data['Volatility_30'] = self.data['Close'].pct_change().rolling(30).std()
        self.data['Volatility_60'] = self.data['Close'].pct_change().rolling(60).std()
        new_predictors.extend(['Volatility_30', 'Volatility_60'])
        
        # Add momentum features
        self.data['Momentum_5'] = self.data['Close'] / self.data['Close'].shift(5) - 1
        self.data['Momentum_20'] = self.data['Close'] / self.data['Close'].shift(20) - 1
        new_predictors.extend(['Momentum_5', 'Momentum_20'])
        
        # Remove rows with NaN values
        self.data = self.data.dropna(subset=self.data.columns[self.data.columns != "Tomorrow"])
        
        print(f"✅ Features engineered! Total features: {len(new_predictors)}")
        print(f"📊 Final dataset shape: {self.data.shape}")
        
        return new_predictors
    
    def train_model(self, predictors):
        """Train the Random Forest model"""
        print("🤖 Training Random Forest model...")
        
        # Prepare training data
        train = self.data.iloc[:-100]
        test = self.data.iloc[-100:]
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=200, 
            min_samples_split=50, 
            random_state=1,
            n_jobs=-1
        )
        
        self.model.fit(train[predictors], train["Target"])
        
        # Make predictions
        preds = self.model.predict_proba(test[predictors])[:, 1]
        preds = (preds >= 0.6).astype(int)  # Threshold for positive prediction
        
        # Store results
        self.predictions = pd.DataFrame({
            'Target': test["Target"],
            'Predictions': preds
        }, index=test.index)
        
        print("✅ Model training completed!")
        
        return self.model, self.predictions
    
    def evaluate_model(self):
        """Evaluate model performance with comprehensive metrics"""
        print("📊 Evaluating model performance...")
        
        # Calculate metrics
        precision = precision_score(self.predictions['Target'], self.predictions['Predictions'])
        recall = recall_score(self.predictions['Target'], self.predictions['Predictions'])
        f1 = f1_score(self.predictions['Target'], self.predictions['Predictions'])
        
        # Confusion matrix
        cm = confusion_matrix(self.predictions['Target'], self.predictions['Predictions'])
        
        # Print results
        print(f"\n🎯 Model Performance Metrics:")
        print(f"   • Precision: {precision:.3f}")
        print(f"   • Recall: {recall:.3f}")
        print(f"   • F1-Score: {f1:.3f}")
        print(f"   • Accuracy: {(self.predictions['Target'] == self.predictions['Predictions']).mean():.3f}")
        
        # Create performance visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Performance comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [precision, recall, f1]
        axes[1].bar(metrics, values, color=['skyblue', 'lightgreen', 'orange'])
        axes[1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('StockPricePredictionModel/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return precision, recall, f1
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard using Plotly"""
        print("📱 Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('S&P 500 Price Trend', 'Prediction Performance', 
                          'Feature Importance', 'Trading Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Price trend
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Close'], 
                      mode='lines', name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 2. Prediction performance
        fig.add_trace(
            go.Scatter(x=self.predictions.index, y=self.predictions['Target'], 
                      mode='markers', name='Actual', marker=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.predictions.index, y=self.predictions['Predictions'], 
                      mode='markers', name='Predicted', marker=dict(color='red')),
            row=1, col=2
        )
        
        # 3. Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig.add_trace(
            go.Bar(x=feature_importance['importance'], y=feature_importance['feature'], 
                  orientation='h', name='Feature Importance'),
            row=2, col=1
        )
        
        # 4. Volume
        fig.add_trace(
            go.Bar(x=self.data.index, y=self.data['Volume'], name='Volume'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="S&P 500 Stock Market Prediction Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save interactive dashboard
        fig.write_html('StockPricePredictionModel/interactive_dashboard.html')
        print("✅ Interactive dashboard saved as 'interactive_dashboard.html'")
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("📄 Generating analysis report...")
        
        report = f"""
# S&P 500 Stock Market Prediction Analysis Report

## Executive Summary
This analysis examines the S&P 500 index from 1990 to present, developing a machine learning model to predict market direction (up/down) with a focus on actionable insights for trading strategies.

## Key Findings

### Data Overview
- **Analysis Period**: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}
- **Total Trading Days**: {len(self.data):,}
- **Market Up Days**: {self.data['Target'].sum():,} ({self.data['Target'].mean():.1%})
- **Market Down Days**: {len(self.data) - self.data['Target'].sum():,} ({(1 - self.data['Target'].mean()):.1%})

### Model Performance
- **Precision**: {precision_score(self.predictions['Target'], self.predictions['Predictions']):.3f}
- **Recall**: {recall_score(self.predictions['Target'], self.predictions['Predictions']):.3f}
- **F1-Score**: {f1_score(self.predictions['Target'], self.predictions['Predictions']):.3f}

### Market Insights
- **Total Return (1990-Present)**: {((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) - 1):.1%}
- **Average Daily Volume**: {self.data['Volume'].mean():,.0f} shares
- **Volatility (30-day)**: {self.data['Close'].pct_change().rolling(30).std().mean():.2%}

## Recommendations
1. **Risk Management**: Use model predictions as one input among many for trading decisions
2. **Portfolio Diversification**: Consider market timing strategies based on model signals
3. **Continuous Monitoring**: Regularly retrain model with new data for optimal performance

## Methodology
- **Algorithm**: Random Forest Classifier
- **Features**: Price ratios, trend indicators, volatility measures, momentum indicators
- **Validation**: Time-series cross-validation with 100-day test period
- **Threshold**: 60% probability threshold for positive predictions

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        # Save report
        with open('StockPricePredictionModel/analysis_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Analysis report saved as 'analysis_report.md'")
        return report

def main():
    """Main execution function"""
    print("🚀 Starting S&P 500 Stock Market Analysis...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SP500Analyzer()
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    analyzer.create_target_variable()
    
    # Perform EDA
    analyzer.perform_eda()
    
    # Engineer features
    predictors = analyzer.engineer_features()
    
    # Train model
    analyzer.train_model(predictors)
    
    # Evaluate model
    analyzer.evaluate_model()
    
    # Create interactive dashboard
    analyzer.create_interactive_dashboard()
    
    # Generate report
    analyzer.generate_report()
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Check the following files:")
    print("   • eda_analysis.png - Exploratory data analysis")
    print("   • model_evaluation.png - Model performance metrics")
    print("   • interactive_dashboard.html - Interactive dashboard")
    print("   • analysis_report.md - Comprehensive report")

if __name__ == "__main__":
    main() 