"""
🏠 REAL ESTATE INVESTMENT ADVISOR - PROFESSIONAL EDITION
AI-Driven Forecasting | Robust ML Scoring | Focused Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================
# PAGE CONFIGURATION - PROFESSIONAL
# ============================================
st.set_page_config(
    page_title="🏠 Pro Real Estate Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - ULTRA PROFESSIONAL DESIGN (Refined)
# ============================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    
    /* Main Header - Company Style Interface */
    .main-header-container {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); /* Dark Blue/Slate */
        border-radius: 0 0 30px 30px;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #e2e8f0 !important;
        margin-bottom: 10px !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    
    .sub-header {
        font-size: 1.3rem !important;
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }
    
    /* Cards - Elegant */
    .professional-card {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        margin: 15px 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s;
    }
    
    .professional-card:hover {
        box-shadow: 0 15px 45px rgba(59, 130, 246, 0.1);
    }
    
    /* Investment Status - Highlighting Key Metrics */
    .investment-status-excellent, .investment-status-good, .investment-status-fair, .investment-status-poor {
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
        margin: 15px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .investment-status-excellent {
        background: #d1fae5; color: #065f46; border: 2px solid #10b981;
    }
    .investment-status-good {
        background: #dbeafe; color: #1e40af; border: 2px solid #3b82f6;
    }
    .investment-status-fair {
        background: #fef3c7; color: #92400e; border: 2px solid #f59e0b;
    }
    .investment-status-poor {
        background: #fee2e2; color: #991b1b; border: 2px solid #ef4444;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3b82f6;
        display: inline-block;
        padding-right: 50px;
    }
    
    /* Metric Score Visualization */
    .score-container {
        padding: 20px;
        border-radius: 12px;
        background: #f0f4f8;
        margin-top: 20px;
        text-align: center;
    }
    .final-score {
        font-size: 3rem;
        font-weight: 900;
        color: #3b82f6;
        margin: 10px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a; /* Same as header for consistency */
    }
    [data-testid="stSidebar"] .stButton button {
        background: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Info Box */
    .prediction-info {
        background: #fff3e0;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA MANAGER - ADVANCED ML PREDICTIONS (Prediction Logic)
# ============================================
class AdvancedMLPredictor:
    """
    Advanced Investment Prediction Engine using ML principles.
    Combines Rule-based Heuristics with a simulated Random Forest Regressor.
    """
    
    def __init__(self):
        self.market_data = self._load_market_data()
        self.property_types = ['Apartment', 'Villa', 'Independent House', 'Penthouse', 
                               'Builder Floor', 'Studio', 'Farm House', 'Flat']
        self.le = LabelEncoder()
        # Initialize and "train" a dummy ML model
        self.ml_model = self._setup_ml_model()

    def _load_market_data(self):
        """Load comprehensive market data for 8 major Indian cities."""
        return {
            'Mumbai': {'avg_price': 350, 'growth': 8.5, 'demand': 'Very High', 'rental_yield': 3.2, 'infrastructure': 9.0, 'job_growth': 8.8, 'price_per_sqft': 23000, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5, 5: 2.0}},
            'Delhi': {'avg_price': 220, 'growth': 7.2, 'demand': 'High', 'rental_yield': 2.8, 'infrastructure': 8.5, 'job_growth': 7.5, 'price_per_sqft': 18000, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.4, 5: 1.8}},
            'Bangalore': {'avg_price': 180, 'growth': 9.1, 'demand': 'Very High', 'rental_yield': 3.5, 'infrastructure': 8.8, 'job_growth': 9.2, 'price_per_sqft': 15000, 'bhk_multiplier': {1: 0.9, 2: 1.0, 3: 1.3, 4: 1.6, 5: 2.0}},
            'Hyderabad': {'avg_price': 150, 'growth': 8.2, 'demand': 'High', 'rental_yield': 3.0, 'infrastructure': 8.0, 'job_growth': 8.5, 'price_per_sqft': 12000, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.4, 5: 1.7}},
            'Pune': {'avg_price': 130, 'growth': 7.5, 'demand': 'High', 'rental_yield': 2.9, 'infrastructure': 7.8, 'job_growth': 7.8, 'price_per_sqft': 11000, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.4, 5: 1.7}},
            'Chennai': {'avg_price': 120, 'growth': 6.8, 'demand': 'Medium', 'rental_yield': 2.5, 'infrastructure': 7.5, 'job_growth': 6.5, 'price_per_sqft': 10000, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.1, 4: 1.3, 5: 1.6}},
            'Kolkata': {'avg_price': 100, 'growth': 5.5, 'demand': 'Medium', 'rental_yield': 2.3, 'infrastructure': 7.0, 'job_growth': 5.8, 'price_per_sqft': 8500, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.1, 4: 1.3, 5: 1.5}},
            'Ahmedabad': {'avg_price': 110, 'growth': 6.5, 'demand': 'Medium', 'rental_yield': 2.6, 'infrastructure': 7.2, 'job_growth': 6.2, 'price_per_sqft': 9000, 'bhk_multiplier': {1: 0.8, 2: 1.0, 3: 1.1, 4: 1.3, 5: 1.6}}
        }
        
    def _create_synthetic_data(self):
        """Generates synthetic data for ML model training simulation."""
        N = 500
        all_labels = list(self.market_data.keys()) + self.property_types
        self.le.fit(all_labels)

        data = pd.DataFrame({
            'city_enc': self.le.transform(np.random.choice(list(self.market_data.keys()), N)),
            'type_enc': self.le.transform(np.random.choice(self.property_types, N)),
            'bhk': np.random.randint(1, 6, N),
            'size_sqft': np.random.randint(500, 5000, N),
            'age': np.random.randint(0, 30, N),
            'amenities_score': np.random.uniform(5, 10, N).round(1),
            'growth': np.random.uniform(5, 10, N).round(1) 
        })
        
        # Target variable: Simulate a final investment score (0-100)
        data['score'] = (
            (data['growth'] * 5) + 
            (data['amenities_score'] * 4) + 
            (data['size_sqft'] / 100) + 
            (100 - data['age'] * 1.5)
        ) + np.random.normal(0, 10, N)
        data['score'] = np.clip(data['score'] - data['score'].min(), 0, 100).round(0)
        return data

    def _setup_ml_model(self):
        """Sets up and 'trains' the Random Forest model (simulation)."""
        try:
            synthetic_df = self._create_synthetic_data()
            X = synthetic_df[['city_enc', 'type_enc', 'bhk', 'size_sqft', 'age', 'amenities_score', 'growth']]
            y = synthetic_df['score']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            return model
        except Exception as e:
            st.error(f"ML Model Setup Error: Could not initialize model. Using fallback scores. (Details: {e})")
            return None

    def _get_ml_prediction(self, property_data):
        """Gets a prediction score from the ML model (0-100)."""
        try:
            city_enc = self.le.transform([property_data['city']])[0]
            type_enc = self.le.transform([property_data['property_type']])[0]
        except ValueError:
            return np.random.uniform(50, 70) 

        features = pd.DataFrame([[
            city_enc,
            type_enc,
            property_data['bhk'],
            property_data['size_sqft'],
            property_data['age'],
            property_data['amenities_score'],
            self.market_data.get(property_data['city'], {}).get('growth', 7.0)
        ]], columns=['city_enc', 'type_enc', 'bhk', 'size_sqft', 'age', 'amenities_score', 'growth'])
        
        ml_score = self.ml_model.predict(features)[0]
        return np.clip(ml_score, 0, 100)
    
    def predict_investment(self, property_data):
        """
        Professional Investment Prediction: Combines Rule-based Heuristics 
        with ML Model Score using a weighted average (ML=70%, Heuristic=30%).
        """
        start_time = time.time()
        
        # 1. ML Model Prediction
        if self.ml_model:
            ml_score = self._get_ml_prediction(property_data)
        else:
            ml_score = 60 # Fallback score

        # 2. Rule-based Heuristic Calculation (30% weight)
        city = property_data['city']
        property_type = property_data['property_type']
        bhk = property_data['bhk']
        size = property_data['size_sqft']
        price = property_data['price']
        age = property_data['age']
        amenities = property_data['amenities_score']
        
        city_data = self.market_data.get(city, self.market_data['Bangalore'])
        
        price_per_sqft = city_data['price_per_sqft']
        bhk_multiplier = city_data['bhk_multiplier'].get(bhk, 1.0)
        type_multiplier = {
            'Apartment': 1.0, 'Flat': 1.0, 'Villa': 1.5, 'Independent House': 1.4, 
            'Penthouse': 1.8, 'Builder Floor': 1.1, 'Studio': 0.8, 'Farm House': 1.3
        }.get(property_type, 1.0)
        
        fair_price = (size * price_per_sqft * bhk_multiplier * type_multiplier) / 100000
        price_ratio = price / fair_price if fair_price > 0 else 1
        
        heuristic_score = 0
        
        # Scoring Components (Total 100 points based on rules)
        if price_ratio < 0.9: heuristic_score += 30
        elif price_ratio < 1.1: heuristic_score += 20
        elif price_ratio < 1.3: heuristic_score += 10
        
        demand_points = {'Very High': 25, 'High': 20, 'Medium': 15, 'Low': 10}.get(city_data['demand'], 15)
        heuristic_score += demand_points
        
        if age <= 5: heuristic_score += 20
        elif age <= 10: heuristic_score += 15
        elif age <= 20: heuristic_score += 10
        else: heuristic_score += 5
        
        heuristic_score += (amenities / 10) * 15
        heuristic_score += (city_data['growth'] / 10) * 10
        
        heuristic_score = min(heuristic_score, 100)

        # 3. Final Professional Score (Weighted Average)
        final_score = (ml_score * 0.7) + (heuristic_score * 0.3)
        score = min(final_score, 100) 
        
        # 4. Determine Investment Status (How it works: Score is High, Fair, or Low)
        
        # HIGH SCORE (88-100): Both ML and Heuristic models strongly agree on high investment quality. 
        # Indicates undervalued property and high market growth potential.
        if score >= 88:
            status = "EXCELLENT INVESTMENT (HIGH SCORE)"
            status_class = "investment-status-excellent"
        
        # GOOD SCORE (75-87): Good agreement between models. Indicates fair valuation, solid market fundamentals, 
        # and strong returns potential. This is typically the 'Buy' zone.
        elif score >= 75:
            status = "STRONG INVESTMENT POTENTIAL (GOOD SCORE)"
            status_class = "investment-status-good"
        
        # FAIR SCORE (60-74): Mixed signals. Either the price is slightly high (Heuristic Low) 
        # or the location has moderate growth (ML Moderate). Recommendation: Monitor or proceed with caution.
        elif score >= 60:
            status = "FAIR OPPORTUNITY - MONITOR (MODERATE SCORE)"
            status_class = "investment-status-fair"
        
        # LOW SCORE (<60): Both models or the highly-weighted ML model disagrees with the investment. 
        # Indicates significant overvaluation or poor long-term growth prospects.
        else:
            status = "HIGH RISK / POOR INVESTMENT (LOW SCORE)"
            status_class = "investment-status-poor"
            
        # 5. Financial Forecast
        growth_rate = city_data['growth'] / 100
        future_price = price * ((1 + growth_rate) ** 5)
        annual_appreciation = growth_rate * 100
        
        variation = np.random.normal(0, future_price * 0.05)
        future_price = max(future_price + variation, price * 1.1)
        
        prediction_time = time.time() - start_time
        
        return {
            'score': score,
            'status': status,
            'status_class': status_class,
            'future_price': future_price,
            'annual_appreciation': annual_appreciation,
            'fair_price': fair_price,
            'price_valuation': 'Undervalued' if price_ratio < 0.95 else 'Fairly Valued' if price_ratio < 1.05 else 'Overvalued',
            'prediction_time': prediction_time,
            'city_data': city_data,
            'ml_score': ml_score,           
            'heuristic_score': heuristic_score 
        }

# ============================================
# MAIN APPLICATION CLASS
# ============================================
class RealEstateAdvisorPro:
    def __init__(self):
        self.predictor = AdvancedMLPredictor()
        # Removed properties loading and management since 'property search' is removed
        self.properties = pd.DataFrame() 
        
    def show_header(self):
        """Show prominent header."""
        st.markdown("""
        <div class="main-header-container">
            <h1 class="main-header">🏠 ELITE INVESTMENT FORECASTING SYSTEM</h1>
            <h2 class="sub-header">AI-Powered Due Diligence for High-Value Real Estate Assets</h2>
        </div>
        """, unsafe_allow_html=True)
    
    def show_sidebar(self):
        """Show professional sidebar with navigation."""
        with st.sidebar:
            st.markdown("### 🧭 APPLICATION MODULES")
            
            nav_options = {
                "📈 Executive Dashboard": "dashboard",
                "🤖 AI Predictor (Main Tool)": "predictor",
                "📊 Market Analysis": "market",
                "❓ Prediction Logic": "logic", # New Page for logic explanation
                "⚙️ Technical Details": "skills",
            }
            
            # Custom navigation buttons
            for option, key in nav_options.items():
                if st.button(option, use_container_width=True, key=f"nav_{key}"):
                    st.session_state.current_page = key
            
            st.markdown("---")
            st.markdown("### ⚡ SYSTEM STATUS")
            st.info(f"Model Engine: Random Forest (70% Weight)")
            st.info(f"Heuristic Logic: Active (30% Weight)")
            st.info(f"Market Coverage: {len(self.predictor.market_data)} Major Cities")

    def show_dashboard(self):
        """Show the main dashboard with key metrics and charts."""
        st.markdown('<h2 class="section-header">📈 EXECUTIVE DASHBOARD</h2>', unsafe_allow_html=True)
        
        # Key Metrics (Using columns and custom styling for cleanliness)
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate key averages
        avg_growth = np.mean([data['growth'] for data in self.predictor.market_data.values()])
        avg_rental = np.mean([data['rental_yield'] for data in self.predictor.market_data.values()])
        top_city = max(self.predictor.market_data.items(), key=lambda x: x[1]['growth'])
        
        metrics = [
            ("Avg Growth Rate", f"{avg_growth:.1f}%", "Annual Appreciation", "#10b981"),
            ("Avg Rental Yield", f"{avg_rental:.1f}%", "Market Potential", "#f59e0b"),
            ("Market Coverage", f"{len(self.predictor.market_data)} Cities", "India Metro Focus", "#3b82f6"),
            ("Top City (Growth)", f"{top_city[0]}", f"{top_city[1]['growth']}%", "#8b5cf6")
        ]
        
        for i, (label, value, delta, color) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div class="professional-card" style="border-left: 5px solid {color}; padding: 20px;">
                    <div style='font-size: 0.8rem; color: #64748b; font-weight: 600;'>{label}</div>
                    <div style='font-size: 2rem; font-weight: 800; color: #1e293b; margin: 5px 0;'>{value}</div>
                    <div style='font-size: 0.8rem; color: {color}; font-weight: 600;'>{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Market Analysis Charts
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Market Metrics Comparison")
        
        cities = list(self.predictor.market_data.keys())
        comparison_data = [
            {'City': c, 'Growth': self.predictor.market_data[c]['growth'], 'Yield': self.predictor.market_data[c]['rental_yield'], 'Price': self.predictor.market_data[c]['avg_price']}
            for c in cities
        ]
        comparison_df = pd.DataFrame(comparison_data)

        col_left, col_right = st.columns(2)
        
        with col_left:
            fig_growth = px.bar(
                comparison_df.sort_values('Growth', ascending=False),
                x='City', y='Growth', color='Growth',
                color_continuous_scale=px.colors.sequential.Teal,
                title="City Growth Rate (%)", height=400
            )
            st.plotly_chart(fig_growth, use_container_width=True)
            
        with col_right:
            fig_yield = px.scatter(
                comparison_df,
                x='Growth', y='Yield', size='Price', color='City',
                title="Growth vs. Rental Yield (Bubble Size = Avg Price)", height=400
            )
            st.plotly_chart(fig_yield, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Removed show_property_search
    
    def show_ai_predictor(self):
        """Show the interactive AI prediction interface."""
        st.markdown('<h2 class="section-header">🤖 AI INVESTMENT PREDICTOR</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-card" style="border-left: 5px solid #3b82f6;">
            <h3>🔮 Real-Time Investment Due Diligence</h3>
            <p style="color: #64748b;">
                Enter property details to receive a dual-layered investment assessment: 
                a **Machine Learning Score** (70% weight) and a **Heuristic Score** (30% weight) 
                for maximum prediction confidence.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction Form
        col_input, col_output = st.columns([1, 1])
        
        with col_input:
            st.markdown('<div class="professional-card">', unsafe_allow_html=True)
            st.markdown("### 📝 Property Inputs")
            
            city = st.selectbox("City", options=list(self.predictor.market_data.keys()), index=2, key="pr_city")
            col1, col2 = st.columns(2)
            with col1: property_type = st.selectbox("Property Type", options=self.predictor.property_types, index=0, key="pr_type")
            with col2: bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=1, key="pr_bhk")
            
            size_sqft = st.number_input("Size (Square Feet)", min_value=100, max_value=10000, value=1200, step=100, key="pr_size")
            price = st.number_input("Current Price (₹ Lakhs)", min_value=10, max_value=10000, value=150, step=10, key="pr_price")
            age = st.slider("Property Age (Years)", min_value=0, max_value=50, value=5, key="pr_age")
            amenities_score = st.slider("Amenities Score (1-10)", min_value=1, max_value=10, value=7, key="pr_amenities")
            
            predict_clicked = st.button("🚀 EXECUTE AI FORECAST", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_output:
            if predict_clicked:
                property_data = {
                    'city': city, 'property_type': property_type, 'bhk': bhk, 'size_sqft': size_sqft, 
                    'price': price, 'age': age, 'amenities_score': amenities_score
                }
                
                with st.spinner("🤖 AI is running dual-layered analysis and forecasting..."):
                    time.sleep(1) 
                    prediction = self.predictor.predict_investment(property_data)
                
                st.markdown(f'<div class="{prediction["status_class"]}"> {prediction["status"]} </div>', unsafe_allow_html=True)
                
                # Overall Score Gauge
                st.markdown(f"""
                <div class="score-container">
                    <div style='font-size: 0.9rem; font-weight: 600; color: #1e293b;'>COMPOSITE INVESTMENT SCORE</div>
                    <div class="final-score" style="color: #1e40af;">{prediction['score']:.0f}/100</div>
                    <p style='font-weight: 700; color: #10b981;'>{prediction['price_valuation']} Value</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Breakdown and Forecast
                st.markdown('<div class="professional-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Prediction Breakdown")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("ML Model Score (70% Weight)", f"{prediction['ml_score']:.1f}")
                    st.metric("Heuristic Score (30% Weight)", f"{prediction['heuristic_score']:.1f}")
                    st.metric("Current Fair Price", f"₹{prediction['fair_price']:.1f} L")
                with col_b:
                    st.metric("5-Year Price Forecast", f"₹{prediction['future_price']:.1f} L", f"+{((prediction['future_price']/price)-1)*100:.1f}% Total")
                    st.metric("Annual Growth (CAGR)", f"{prediction['annual_appreciation']:.1f}%")
                    st.metric("City Demand Level", prediction['city_data']['demand'])
                    
                st.markdown('---')
                st.markdown("### 🎯 Professional Recommendation")
                
                recommendations = []
                if prediction['score'] >= 88: recommendations = ["✅ **Strong Buy:** Excellent alignment of value, growth, and ML confirmation.", "✅ Property is likely undervalued or perfectly priced for high growth.", "✅ Proceed with due diligence quickly."]
                elif prediction['score'] >= 75: recommendations = ["👍 **Buy:** Solid potential, monitor local competition and specific locality data.", "👍 Expect consistent returns; negotiating price is still advisable.", "👍 Good long-term holding asset."]
                elif prediction['score'] >= 60: recommendations = ["⚠️ **Hold/Re-evaluate:** Fair opportunity, but check for specific risks (age, location quality).", "⚠️ May be slightly overvalued by the market.", "⚠️ Explore lower-priced alternatives."]
                else: recommendations = ["❌ **Avoid:** High risk due to poor valuation or weak market fundamentals.", "❌ Wait for significant price correction (15%+).", "❌ Re-allocate capital to higher-scoring cities."]
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def show_market_analysis(self):
        """Show comprehensive market analysis."""
        st.markdown('<h2 class="section-header">📊 MARKET ANALYSIS</h2>', unsafe_allow_html=True)
        
        # Market Overview
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 🏙️ City Comparison Analysis")
        
        comparison_data = []
        for city, data in self.predictor.market_data.items():
            comparison_data.append({
                'City': city, 'Avg Price (₹L)': data['avg_price'], 'Growth Rate (%)': data['growth'], 
                'Demand': data['demand'], 'Rental Yield (%)': data['rental_yield'], 
                'Infrastructure (/10)': data['infrastructure'], 'Job Growth (%)': data['job_growth']
            })
        comparison_df = pd.DataFrame(comparison_data)
        
        cols = st.columns(6)
        metrics = ['Avg Price (₹L)', 'Growth Rate (%)', 'Demand', 'Rental Yield (%)', 'Infrastructure (/10)', 'Job Growth (%)']
        
        for idx, metric in enumerate(metrics):
            with cols[idx]:
                if metric == 'Demand':
                    top_city = comparison_df.loc[comparison_df[metric].isin(['Very High', 'High'])].iloc[0]
                else:
                    top_city = comparison_df.loc[comparison_df[metric].idxmax()]
                st.metric(metric.split('(')[0].strip(), f"{top_city[metric]}", top_city['City'])
        
        # Interactive chart
        col1, col2 = st.columns(2)
        with col1: x_axis = st.selectbox("X-Axis Metric", options=['Avg Price (₹L)', 'Growth Rate (%)', 'Rental Yield (%)', 'Infrastructure (/10)', 'Job Growth (%)'], index=0)
        with col2: y_axis = st.selectbox("Y-Axis Metric", options=['Growth Rate (%)', 'Avg Price (₹L)', 'Rental Yield (%)', 'Infrastructure (/10)', 'Job Growth (%)'], index=1)
        
        fig = px.scatter(
            comparison_df, x=x_axis, y=y_axis, size='Avg Price (₹L)', color='Demand', text='City',
            title=f"{x_axis} vs {y_axis}", color_continuous_scale='Viridis', size_max=60
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Price Trends
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Price Trends & Forecast (2020-2028)")
        
        years = list(range(2020, 2029))
        fig = go.Figure()
        
        for city, data in self.predictor.market_data.items():
            prices = []
            growth_rate = data['growth'] / 100
            for year in years:
                year_diff = abs(year - 2024)
                if year <= 2024:
                    price = data['avg_price'] / ((1 + growth_rate) ** year_diff)
                else:
                    price = data['avg_price'] * ((1 + growth_rate) ** year_diff)
                prices.append(price)
            
            fig.add_trace(go.Scatter(x=years, y=prices, mode='lines+markers', name=city, line=dict(width=3)))
        
        fig.update_layout(title="Historical & Forecast Price Trends", xaxis_title="Year", yaxis_title="Price (₹ Lakhs)", height=500, hovermode="x unified", template="plotly_white", showlegend=True)
        fig.add_vline(x=2024, line_dash="dash", line_color="red", annotation_text="Current Year")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def show_technical_skills(self):
        """Show technical skills and project architecture."""
        st.markdown('<h2 class="section-header">⚙️ TECHNICAL STACK & ARCHITECTURE</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Technology Stack Used")
        
        tech_stack = {
            "Programming": {"Python": 95, "Pandas": 90, "NumPy": 85},
            "Machine Learning": {"Random Forest (Reg.)": 90, "Scikit-learn": 88, "Feature Engineering": 85},
            "Web Framework": {"Streamlit": 95, "Interactive UI": 90, "Custom CSS": 85},
            "Data Visualization": {"Plotly": 92, "Custom Charts": 80},
        }
        
        cols = st.columns(2)
        col_idx = 0
        
        for category, skills in tech_stack.items():
            with cols[col_idx % 2]:
                st.markdown(f"#### 🔧 {category}")
                for skill, proficiency in skills.items():
                    st.markdown(f"""
                    <div style='margin: 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                            <span style='font-weight: 600;'>{skill}</span>
                            <span style='color: #3b82f6; font-weight: 700;'>{proficiency}%</span>
                        </div>
                        <div style='height: 8px; background: #e2e8f0; border-radius: 4px;'>
                            <div style='height: 100%; width: {proficiency}%; background: linear-gradient(90deg, #3b82f6, #1e40af); border-radius: 4px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            col_idx += 1
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 🏗️ Prediction Model Architecture")
        st.code("""
┌──────────────────────────────────────┐
│        USER INPUTS (Property Data)   │
└─────────────────────┬────────────────┘
                      │
┌─────────────────────▼────────────────┐
│      ADVANCED ML PREDICTOR LAYER     │
│  (1) Machine Learning Model (70% W)  │
│  (2) Heuristic Scoring Logic (30% W) │
└─────────────────────┬────────────────┘
                      │
┌─────────────────────▼────────────────┐
│      FINAL COMPOSITE INVESTMENT SCORE│
│  (0-100) -> HIGH, GOOD, FAIR, or LOW │
└──────────────────────────────────────┘
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    def show_prediction_logic(self):
        """Explain how the ML prediction works (High/Low/Fair)."""
        st.markdown('<h2 class="section-header">❓ PREDICTION LOGIC & INTERPRETATION</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="professional-card prediction-info">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Score कैसे काम करता है?")
        st.markdown("""
        हमारा **कंपोजिट इन्वेस्टमेंट स्कोर (Composite Investment Score)** 100 में से एक वैल्यू है, जो दो अलग-अलग मॉडलों के आउटपुट को मिलाकर बनता है:
        
        1.  **Machine Learning Score (70% Weight):** यह एक `RandomForestRegressor` मॉडल का आउटपुट है। यह मॉडल **ऐतिहासिक डेटा (simulated historical data)** के आधार पर देखता है कि आपके इनपुट (City, Size, Age, Amenities) वाली प्रॉपर्टी ने कितना बेहतर प्रदर्शन किया होगा। यह स्कोर **पैटर्न और जटिल संबंधों** पर केंद्रित होता है।
        
        2.  **Heuristic Score (30% Weight):** यह **नियम-आधारित (Rule-based)** स्कोर है। यह सीधे नियमों पर आधारित है, जैसे:
            * अगर प्रॉपर्टी **बाजार मूल्य से सस्ती (Undervalued)** है, तो स्कोर बढ़ता है।
            * अगर प्रॉपर्टी **नई** है (Age कम), तो स्कोर बढ़ता है।
            * अगर सिटी की **ग्रोथ रेट (Growth Rate)** और **डिमांड (Demand)** ज़्यादा है, तो स्कोर बढ़ता है।
        
        फाइनल स्कोर इन दोनों को मिलाकर मिलता है, जिससे प्रेडिक्शन **ज़्यादा स्थिर (stable) और विश्वसनीय (reliable)** होता है।
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 📊 स्कोर इंटरप्रिटेशन (High/Low/Fair)")
        st.table(pd.DataFrame({
            'Score Range': ['88 - 100', '75 - 87', '60 - 74', '0 - 59'],
            'Investment Status': ['**EXCELLENT (HIGH SCORE)**', '**STRONG (GOOD SCORE)**', '**FAIR (MODERATE SCORE)**', '**POOR (LOW SCORE)**'],
            'Interpretation': [
                '**Strong Buy Signal:** ML model confirms excellent value and high market momentum. **Highly recommended.**',
                '**Solid Buy Signal:** Good property, fair price, and healthy growth in the area. Consistent long-term potential.',
                '**Cautionary Hold:** Mixed signals. Property may be slightly overvalued or growth potential is moderate. Monitor market.',
                '**Avoid/High Risk:** Significant overvaluation, high age, or weak city fundamentals. Price negotiation necessary.'
            ],
            'Key Focus': ['**Undervaluation + High Growth**', 'Fair Price + Solid Growth', 'Market Risk/Moderate Value', 'Overvaluation + Low Demand']
        }))
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_project_details(self):
        """Show about project section (Final section after logic)."""
        self.show_prediction_logic()
        
        st.markdown('<h2 class="section-header">ℹ️ PROJECT DETAILS</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Project Goal and Capabilities")
        st.markdown("""
        The **Real Estate AI Advisor Pro** is a data science demonstration project built on Streamlit, 
        designed to mimic a professional property analysis tool.
        
        ### ✨ Core Capabilities:
        
        * **ML-Enhanced Scoring:** Utilizes a simulated Random Forest model combined with weighted heuristics for highly stable investment scores.
        * **Future Forecasting:** Provides 5-year price projections based on historical city growth rates.
        * **Professional UI/UX:** Features a clean, focused design for optimal clarity and user experience.
        
        ---
        
        ### ⚖️ Disclaimer (ज़रूरी सूचना)
        
        यह एप्लीकेशन केवल **शैक्षणिक और प्रदर्शन (educational and demonstrational)** उद्देश्यों के लिए बनाया गया है। इसमें उपयोग किया गया Machine Learning मॉडल **सिम्युलेटेड डेटा** पर आधारित है।
        
        **वास्तविक निवेश निर्णय लेने से पहले हमेशा किसी पेशेवर वित्तीय सलाहकार (professional financial advisor)** से परामर्श लें।
        
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MAIN EXECUTION LOGIC
# ============================================
def main():
    # Initialize app state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    app = RealEstateAdvisorPro()
    
    # 1. Show Header
    app.show_header()
    
    # 2. Show Sidebar (handles navigation buttons)
    app.show_sidebar()
    
    # 3. Show Main Content based on navigation state
    st.markdown('<div style="padding: 0 20px;">', unsafe_allow_html=True)
    
    if st.session_state.current_page == 'dashboard':
        app.show_dashboard()
    elif st.session_state.current_page == 'market':
        app.show_market_analysis()
    elif st.session_state.current_page == 'predictor':
        app.show_ai_predictor()
    elif st.session_state.current_page == 'skills':
        app.show_technical_skills()
    # Combine logic and details pages
    elif st.session_state.current_page == 'logic':
        app.show_prediction_logic()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
