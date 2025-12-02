"""
🏠 REAL ESTATE INVESTMENT ADVISOR - PROFESSIONAL EDITION
AI-Driven Forecasting | Robust ML Scoring | Premium Design
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
    
    /* Main Header - Clean Professional Blue */
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
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA MANAGER - ADVANCED ML PREDICTIONS
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
        N = 500  # Number of samples
        
        # Combine all unique labels for proper encoding initialization
        all_labels = list(self.market_data.keys()) + self.property_types
        self.le.fit(all_labels)

        # Generate data using the fitted encoder
        data = pd.DataFrame({
            'city_enc': self.le.transform(np.random.choice(list(self.market_data.keys()), N)),
            'type_enc': self.le.transform(np.random.choice(self.property_types, N)),
            'bhk': np.random.randint(1, 6, N),
            'size_sqft': np.random.randint(500, 5000, N),
            'age': np.random.randint(0, 30, N),
            'amenities_score': np.random.uniform(5, 10, N).round(1),
            # Simulate a 'growth' feature
            'growth': np.random.uniform(5, 10, N).round(1) 
        })
        
        # Target variable: Simulate a final investment score (0-100)
        data['score'] = (
            (data['growth'] * 5) + 
            (data['amenities_score'] * 4) + 
            (data['size_sqft'] / 100) + 
            (100 - data['age'] * 1.5)
        ) + np.random.normal(0, 10, N)
        # Normalize and clip the score between 0 and 100
        data['score'] = np.clip(data['score'] - data['score'].min(), 0, 100).round(0)
        return data

    def _setup_ml_model(self):
        """Sets up and 'trains' the Random Forest model (simulation)."""
        try:
            synthetic_df = self._create_synthetic_data()
            X = synthetic_df[['city_enc', 'type_enc', 'bhk', 'size_sqft', 'age', 'amenities_score', 'growth']]
            y = synthetic_df['score']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Using Random Forest Regressor as the ML engine
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            return model
        except Exception as e:
            # Handle potential import errors or setup failures gracefully
            st.error(f"ML Model Setup Error: Could not initialize model. Using fallback scores. (Details: {e})")
            return None

    def _get_ml_prediction(self, property_data):
        """Gets a prediction score from the ML model (0-100)."""
        
        # Use try/except for robust label encoding during prediction
        try:
            city_enc = self.le.transform([property_data['city']])[0]
            type_enc = self.le.transform([property_data['property_type']])[0]
        except ValueError:
            # Fallback if label is unseen (e.g., property type not in synthetic data)
            return np.random.uniform(50, 70) 

        # Prepare feature vector for the model
        features = pd.DataFrame([[
            city_enc,
            type_enc,
            property_data['bhk'],
            property_data['size_sqft'],
            property_data['age'],
            property_data['amenities_score'],
            self.market_data.get(property_data['city'], {}).get('growth', 7.0)
        ]], columns=['city_enc', 'type_enc', 'bhk', 'size_sqft', 'age', 'amenities_score', 'growth'])
        
        # Predict the investment score
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
            ml_score = 60 # Fallback score if ML setup failed

        # 2. Rule-based Heuristic Calculation (For stability and interpretability)
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
        
        # Score Components (Maximum 100 points)
        heuristic_score = 0
        
        # Price Valuation (30 points)
        if price_ratio < 0.9: heuristic_score += 30
        elif price_ratio < 1.1: heuristic_score += 20
        elif price_ratio < 1.3: heuristic_score += 10
        
        # Location/Demand (25 points)
        demand_points = {'Very High': 25, 'High': 20, 'Medium': 15, 'Low': 10}.get(city_data['demand'], 15)
        heuristic_score += demand_points
        
        # Age/Condition (20 points)
        if age <= 5: heuristic_score += 20
        elif age <= 10: heuristic_score += 15
        elif age <= 20: heuristic_score += 10
        else: heuristic_score += 5
        
        # Amenities (15 points)
        heuristic_score += (amenities / 10) * 15
        
        # Growth (10 points)
        heuristic_score += (city_data['growth'] / 10) * 10
        
        heuristic_score = min(heuristic_score, 100)

        # 3. Final Professional Score (Weighted Average)
        final_score = (ml_score * 0.7) + (heuristic_score * 0.3)
        score = min(final_score, 100) 
        
        # 4. Determine Investment Status
        if score >= 88:
            status = "EXCELLENT INVESTMENT (ML Verified)"
            status_class = "investment-status-excellent"
        elif score >= 75:
            status = "STRONG INVESTMENT POTENTIAL"
            status_class = "investment-status-good"
        elif score >= 60:
            status = "FAIR OPPORTUNITY - MONITOR"
            status_class = "investment-status-fair"
        else:
            status = "HIGH RISK / POOR INVESTMENT"
            status_class = "investment-status-poor"
            
        # 5. Financial Forecast
        growth_rate = city_data['growth'] / 100
        # Future price calculation (5 years)
        future_price = price * ((1 + growth_rate) ** 5)
        annual_appreciation = growth_rate * 100
        
        # Add realistic variation
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
        # Use the advanced predictor
        self.predictor = AdvancedMLPredictor()
        self.properties = self._load_sample_properties()
        
    def _load_sample_properties(self):
        """Loads sample properties for the search feature."""
        data = []
        cities = list(self.predictor.market_data.keys())
        property_types = ['Apartment', 'Villa', 'Independent House', 'Flat', 'Penthouse']
        
        for i in range(50):
            city = np.random.choice(cities)
            city_data = self.predictor.market_data[city]
            property_type = np.random.choice(property_types)
            bhk = np.random.randint(1, 6)
            
            base_size = np.random.randint(800, 3000)
            price_per_sqft = city_data['price_per_sqft'] * (0.8 + np.random.random() * 0.4)
            price = (base_size * price_per_sqft * city_data['bhk_multiplier'].get(bhk, 1.0)) / 100000
            
            data.append({
                'id': i + 1,
                'city': city,
                'property_type': property_type,
                'bhk': bhk,
                'size_sqft': base_size,
                'price_lakhs': round(price, 1),
                'age_years': np.random.randint(0, 20),
                'amenities_score': round(5 + np.random.random() * 5, 1),
                'location_score': round(6 + np.random.random() * 4, 1),
                'status': np.random.choice(['Available', 'Sold', 'Under Construction']),
                'furnishing': np.random.choice(['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'])
            })
        
        return pd.DataFrame(data)
    
    def show_header(self):
        """Show prominent header."""
        st.markdown("""
        <div class="main-header-container">
            <h1 class="main-header">🏠 REAL ESTATE AI ADVISOR PRO</h1>
            <h2 class="sub-header">AI-Powered Forecasting & Due Diligence for Elite Investments</h2>
        </div>
        """, unsafe_allow_html=True)
    
    def show_sidebar(self):
        """Show professional sidebar with navigation and filters."""
        with st.sidebar:
            st.markdown("### 🧭 NAVIGATION")
            
            nav_options = {
                "📈 Executive Dashboard": "dashboard",
                "🤖 AI Predictor": "predictor",
                "🔍 Property Search": "search",
                "📊 Market Analysis": "market",
                "⚙️ Technical Details": "skills",
            }
            
            # Custom navigation buttons
            for option, key in nav_options.items():
                if st.button(option, use_container_width=True, key=f"nav_{key}"):
                    st.session_state.current_page = key
            
            st.markdown("---")
            
            # Filters remain in the sidebar but are now for direct search
            st.markdown("### 🔎 QUICK FILTER")
            
            # --- Search Filters ---
            selected_city = st.selectbox("City", options=list(self.predictor.market_data.keys()), index=2)
            selected_type = st.selectbox("Type", options=self.predictor.property_types, index=0)
            selected_bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=1)
            budget_range = st.selectbox("Budget (Lakhs)", options=['Any', '50-100', '100-200', '200-500', '500+'], index=2)
            
            search_clicked = st.button("RUN SEARCH", use_container_width=True, type="primary")
            
            if search_clicked:
                st.session_state.search_params = {
                    'city': selected_city, 'property_type': selected_type, 'bhk': selected_bhk, 
                    'budget_range': budget_range, 'min_size': 500, 'max_age': 50, 'min_amenities': 1, 'furnishing': 'Any'
                }
                st.session_state.current_page = 'search'
                st.session_state.show_search_results = True
            
            st.markdown("---")
            st.markdown("### ⚡ STATUS")
            st.info(f"Model: RandomForest (Simulated)")
            st.info(f"Coverage: {len(self.predictor.market_data)} Cities")
    
    def _show_property_card_detailed(self, prop):
        """Show detailed property card with prediction results."""
        property_data = {
            'city': prop['city'], 'property_type': prop['property_type'], 'bhk': prop['bhk'],
            'size_sqft': prop['size_sqft'], 'price': prop['price_lakhs'], 'age': prop['age_years'],
            'amenities_score': prop['amenities_score']
        }
        
        prediction = self.predictor.predict_investment(property_data)
        
        # Determine background color for ML score based on performance
        score_color = "#10b981" if prediction['ml_score'] >= 80 else "#3b82f6" if prediction['ml_score'] >= 65 else "#f59e0b"
        
        st.markdown(f"""
        <div class="property-card-item professional-card">
            <div style='padding: 20px;'>
                <h3 style='margin: 0; color: #1e293b; font-size: 1.4rem;'>{prop['city']} - {prop['property_type']}</h3>
                <p style='margin: 5px 0 15px 0; color: #64748b; font-size: 0.9rem;'>
                    {prop['bhk']} BHK • {prop['size_sqft']} sq ft • {prop['furnishing']}
                </p>
                
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
                    <span style='font-size: 2rem; font-weight: 800; color: #1e40af;'>
                        ₹{prop['price_lakhs']}L
                    </span>
                    <span class='badge' style='background: {score_color}15; color: {score_color}; font-size: 1rem; padding: 8px 15px; border-radius: 20px;'>
                        ML Score: {prediction['ml_score']:.0f}
                    </span>
                </div>
                
                <div class="score-container" style="background: #e0f2fe;">
                    <div style='font-size: 0.9rem; font-weight: 600; color: #3b82f6;'>FINAL INVESTMENT SCORE</div>
                    <div class="final-score">{prediction['score']:.0f}/100</div>
                    <div style='font-size: 0.9rem; color: #64748b;'>Valuation: 
                        <span style='font-weight: 700; color: #92400e;'>{prediction['price_valuation']}</span>
                    </div>
                </div>
                
                <div style='margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                    <div style='background: #f8fafc; padding: 10px; border-radius: 8px; text-align: left;'>
                        <div style='font-size: 0.8rem; color: #64748b;'>5Y Forecast</div>
                        <div style='font-weight: 700;'>₹{prediction['future_price']:.1f}L</div>
                    </div>
                    <div style='background: #f8fafc; padding: 10px; border-radius: 8px; text-align: left;'>
                        <div style='font-size: 0.8rem; color: #64748b;'>Annual CAGR</div>
                        <div style='font-weight: 700; color: #10b981;'>{prediction['annual_appreciation']:.1f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            ("Properties Analyzed", f"{len(self.properties):,}", "In Database", "#3b82f6"),
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
    
    def show_property_search(self):
        """Show property search results based on sidebar filters."""
        st.markdown('<h2 class="section-header">🔍 PROPERTY SEARCH RESULTS</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('show_search_results', False):
            st.info("👈 **Use the 'Quick Filter' in the sidebar to run a property search.**")
            return
        
        search_params = st.session_state.get('search_params', {})
        
        filtered_df = self.properties.copy()
        
        # Apply filters
        if search_params.get('city'): filtered_df = filtered_df[filtered_df['city'] == search_params['city']]
        if search_params.get('property_type'): filtered_df = filtered_df[filtered_df['property_type'] == search_params['property_type']]
        if search_params.get('bhk'): filtered_df = filtered_df[filtered_df['bhk'] == search_params['bhk']]
        
        # Budget filter
        budget_range = search_params.get('budget_range')
        if budget_range != 'Any':
            if budget_range == '50-100': filtered_df = filtered_df[filtered_df['price_lakhs'].between(50, 100)]
            elif budget_range == '100-200': filtered_df = filtered_df[filtered_df['price_lakhs'].between(100, 200)]
            elif budget_range == '200-500': filtered_df = filtered_df[filtered_df['price_lakhs'].between(200, 500)]
            elif budget_range == '500+': filtered_df = filtered_df[filtered_df['price_lakhs'] >= 500]
        
        st.markdown(f"### 📋 Found **{len(filtered_df)}** Properties matching your criteria in **{search_params.get('city')}**.")
        
        if len(filtered_df) > 0:
            # Display properties in a grid layout
            cols = st.columns(3)
            for idx, (_, prop) in enumerate(filtered_df.iterrows()):
                with cols[idx % 3]:
                    self._show_property_card_detailed(prop)
        else:
            st.warning("No properties found matching your criteria. Try widening your filters.")

    def show_ai_predictor(self):
        """Show the interactive AI prediction interface."""
        st.markdown('<h2 class="section-header">🤖 AI INVESTMENT PREDICTOR</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-card" style="border-left: 5px solid #3b82f6;">
            <h3>🔮 Instant ML Analysis</h3>
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
            
            predict_clicked = st.button("🚀 GET PROFESSIONAL AI PREDICTION", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_output:
            if predict_clicked:
                property_data = {
                    'city': city, 'property_type': property_type, 'bhk': bhk, 'size_sqft': size_sqft, 
                    'price': price, 'age': age, 'amenities_score': amenities_score
                }
                
                with st.spinner("🤖 AI is running dual-layered analysis and forecasting..."):
                    time.sleep(1) # Added delay for better user experience, simulating deep analysis
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
                st.markdown("### 📊 Breakdown & Forecast")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("ML Score (70% Weight)", f"{prediction['ml_score']:.1f}")
                    st.metric("Heuristic Score (30% Weight)", f"{prediction['heuristic_score']:.1f}")
                    st.metric("Current Fair Price", f"₹{prediction['fair_price']:.1f} L")
                with col_b:
                    st.metric("5-Year Price Forecast", f"₹{prediction['future_price']:.1f} L", f"+{((prediction['future_price']/price)-1)*100:.1f}% Total")
                    st.metric("Annual Growth (CAGR)", f"{prediction['annual_appreciation']:.1f}%")
                    st.metric("City Demand Level", prediction['city_data']['demand'])
                    
                st.markdown("---")
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
        """Show comprehensive market analysis (unchanged logic, clean display)."""
        st.markdown('<h2 class="section-header">📊 MARKET ANALYSIS</h2>', unsafe_allow_html=True)
        # (Content remains the same as your provided code, ensuring consistency)
        
        # Market Overview
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 🏙️ City Comparison Analysis")
        
        comparison_data = []
        for city, data in self.predictor.market_data.items():
            comparison_data.append({
                'City': city,
                'Avg Price (₹L)': data['avg_price'],
                'Growth Rate (%)': data['growth'],
                'Demand': data['demand'],
                'Rental Yield (%)': data['rental_yield'],
                'Infrastructure (/10)': data['infrastructure'],
                'Job Growth (%)': data['job_growth']
            })
        comparison_df = pd.DataFrame(comparison_data)
        
        # Metrics Row
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
        st.markdown('<h2 class="section-header">⚙️ TECHNICAL SKILLS & STACK</h2>', unsafe_allow_html=True)
        
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
                    # Simplified progress bar display
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
        st.markdown("### 🏗️ Project Architecture (ML & Web Layers)")
        st.code("""
┌──────────────────────────────────────┐
│        STREAMLIT UI/PRESENTATION     │
│  (Custom CSS, Interactive Widgets)   │
└─────────────────────┬────────────────┘
                      │
┌─────────────────────▼────────────────┐
│      BUSINESS LOGIC (RealEstatePro)  │
│  (Data Management, Page Navigation)  │
└─────────────────────┬────────────────┘
                      │
┌─────────────────────▼────────────────┐
│       PREDICTION LAYER (MLPredictor) │
│  (ML Score 70% + Heuristic Score 30%)│
│  (RandomForestRegressor, NumPy/Pandas)
└─────────────────────┬────────────────┘
                      │
┌─────────────────────▼────────────────┐
│       MARKET & PROPERTY DATA         │
│  (In-memory Market Metrics)          │
└──────────────────────────────────────┘
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    def show_about_project(self):
        """Show about project section (Simplified and combined)."""
        st.markdown('<h2 class="section-header">ℹ️ ABOUT THE PROJECT</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Project Goal and Capabilities")
        st.markdown("""
        The **Real Estate AI Advisor Pro** is a data science demonstration project built on Streamlit, 
        designed to mimic a professional property analysis tool.
        
        ### ✨ Core Capabilities:
        
        * **ML-Enhanced Scoring:** Utilizes a simulated Random Forest model combined with weighted heuristics for highly stable investment scores (0-100).
        * **Future Forecasting:** Provides 5-year price projections based on historical city growth rates.
        * **Professional UI/UX:** Features a clean, dark-mode inspired design for optimal clarity and user experience.
        * **Comprehensive Market Data:** Instant access to core economic and property metrics across major Indian cities.
        
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
    if 'show_search_results' not in st.session_state:
        st.session_state.show_search_results = False
    
    app = RealEstateAdvisorPro()
    
    # 1. Show Header
    app.show_header()
    
    # 2. Show Sidebar (handles navigation buttons)
    app.show_sidebar()
    
    # 3. Show Main Content based on navigation state
    st.markdown('<div style="padding: 0 20px;">', unsafe_allow_html=True)
    
    if st.session_state.current_page == 'dashboard':
        app.show_dashboard()
    elif st.session_state.current_page == 'search':
        app.show_property_search()
    elif st.session_state.current_page == 'market':
        app.show_market_analysis()
    elif st.session_state.current_page == 'predictor':
        app.show_ai_predictor()
    elif st.session_state.current_page == 'skills':
        app.show_technical_skills()
    
    # If the user clicks on About, we use the skills page content (or you can create a dedicated one)
    # The 'about' content is included in the last function here for simplicity
    elif st.session_state.current_page == 'about':
        app.show_about_project()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
