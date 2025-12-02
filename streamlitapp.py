"""
🏢 REAL ESTATE INVESTMENT ADVISOR - ULTIMATE PROFESSIONAL VERSION
With Advanced Property Search, Flat Analysis, Professional Design & All Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import json
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="RealEstate Pro | AI Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - ULTRA PROFESSIONAL DESIGN
# ============================================
st.markdown("""
<style>
    /* Modern Professional Theme */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    /* Premium Header */
    .premium-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #1a237e 0%, #283593 25%, #3949ab 50%, #5c6bc0 75%, #7986cb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .sub-header-pro {
        font-size: 1.3rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.3px;
        opacity: 0.9;
    }
    
    /* Premium Cards */
    .premium-card {
        background: linear-gradient(145deg, #ffffff 0%, #fafafa 100%);
        padding: 2.5rem;
        border-radius: 24px;
        margin: 1.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.06);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #3949ab 0%, #5c6bc0 100%);
    }
    
    .premium-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.12);
        border-color: #3949ab;
    }
    
    /* Investment Status - Premium */
    .investment-premium-good {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        color: white;
        padding: 20px 40px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.4rem;
        display: inline-block;
        margin: 1.5rem 0;
        box-shadow: 0 15px 40px rgba(0, 200, 83, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
    }
    
    .investment-premium-good::after {
        content: '✓';
        position: absolute;
        right: 30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2rem;
        opacity: 0.5;
    }
    
    .investment-premium-bad {
        background: linear-gradient(135deg, #ff1744 0%, #f50057 100%);
        color: white;
        padding: 20px 40px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.4rem;
        display: inline-block;
        margin: 1.5rem 0;
        box-shadow: 0 15px 40px rgba(255, 23, 68, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
    }
    
    .investment-premium-bad::after {
        content: '⚠';
        position: absolute;
        right: 30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2rem;
        opacity: 0.5;
    }
    
    /* Premium Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3949ab 0%, #283593 100%);
        color: white;
        font-weight: 700;
        padding: 18px 45px;
        border-radius: 16px;
        border: none;
        width: 100%;
        font-size: 1.1rem;
        letter-spacing: 0.8px;
        box-shadow: 0 15px 35px rgba(57, 73, 171, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: 0.6s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 25px 50px rgba(57, 73, 171, 0.5);
        background: linear-gradient(135deg, #283593 0%, #1a237e 100%);
    }
    
    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: transparent;
        padding: 0 0 1rem 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 65px;
        white-space: pre-wrap;
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        border-radius: 16px;
        padding: 18px 30px;
        font-weight: 700;
        font-size: 1.1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #3949ab;
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(57, 73, 171, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3949ab 0%, #283593 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 15px 40px rgba(57, 73, 171, 0.4) !important;
        transform: translateY(-3px);
    }
    
    /* Premium Input Fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 14px;
        transition: all 0.3s ease;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    
    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus {
        border-color: #3949ab;
        box-shadow: 0 8px 25px rgba(57, 73, 171, 0.15);
        outline: none;
    }
    
    /* Premium Metrics */
    .premium-metric {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #3949ab;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .premium-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.1);
    }
    
    /* Property Cards */
    .property-card-premium {
        background: linear-gradient(145deg, #ffffff 0%, #fafafa 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .property-card-premium:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 60px rgba(0,0,0,0.12);
        border-color: #3949ab;
    }
    
    .property-card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #3949ab 0%, #5c6bc0 100%);
    }
    
    /* Premium Badges */
    .premium-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 5px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .badge-success-premium {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(0, 200, 83, 0.3);
    }
    
    .badge-warning-premium {
        background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(255, 152, 0, 0.3);
    }
    
    .badge-danger-premium {
        background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(244, 67, 54, 0.3);
    }
    
    .badge-info-premium {
        background: linear-gradient(135deg, #2196f3 0%, #42a5f5 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
    }
    
    /* Premium Charts */
    .chart-premium-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.06);
        border: 1px solid #e0e0e0;
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Sidebar Premium */
    .sidebar-premium {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
        color: white;
    }
    
    /* Footer Premium */
    .footer-premium {
        background: linear-gradient(90deg, #1a237e 0%, #283593 100%);
        color: white;
        padding: 3rem;
        border-radius: 30px 30px 0 0;
        margin-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA GENERATOR - REALISTIC PROPERTY DATA
# ============================================
class PropertyDataGenerator:
    """Generates realistic property data for analysis"""
    
    @staticmethod
    def generate_property_data(num_properties=100):
        """Generate comprehensive property data"""
        
        cities = {
            'Mumbai': {'base_price': 350, 'growth': 8.5, 'type_dist': {'Flat': 60, 'Apartment': 25, 'Villa': 10, 'Penthouse': 5}},
            'Delhi': {'base_price': 220, 'growth': 7.2, 'type_dist': {'Flat': 40, 'Apartment': 40, 'Independent House': 15, 'Penthouse': 5}},
            'Bangalore': {'base_price': 180, 'growth': 9.1, 'type_dist': {'Flat': 30, 'Apartment': 50, 'Villa': 15, 'Penthouse': 5}},
            'Hyderabad': {'base_price': 150, 'growth': 8.2, 'type_dist': {'Flat': 35, 'Apartment': 45, 'Independent House': 15, 'Villa': 5}},
            'Pune': {'base_price': 130, 'growth': 7.5, 'type_dist': {'Flat': 40, 'Apartment': 40, 'Independent House': 15, 'Villa': 5}},
            'Chennai': {'base_price': 120, 'growth': 6.8, 'type_dist': {'Flat': 50, 'Apartment': 35, 'Independent House': 10, 'Villa': 5}},
            'Kolkata': {'base_price': 100, 'growth': 5.5, 'type_dist': {'Flat': 60, 'Apartment': 30, 'Independent House': 8, 'Villa': 2}},
            'Ahmedabad': {'base_price': 110, 'growth': 6.5, 'type_dist': {'Flat': 45, 'Apartment': 35, 'Independent House': 15, 'Villa': 5}}
        }
        
        localities = {
            'Mumbai': ['Bandra', 'Andheri', 'Powai', 'Thane', 'Navi Mumbai', 'Worli', 'Malad'],
            'Delhi': ['South Delhi', 'Gurgaon', 'Noida', 'Dwarka', 'Rohini', 'Greater Noida'],
            'Bangalore': ['Whitefield', 'Koramangala', 'Indiranagar', 'Electronic City', 'Sarjapur', 'HSR Layout'],
            'Hyderabad': ['Gachibowli', 'Hitech City', 'Banjara Hills', 'Jubilee Hills', 'Kondapur'],
            'Pune': ['Hinjewadi', 'Kharadi', 'Viman Nagar', 'Baner', 'Wakad', 'Kalyani Nagar'],
            'Chennai': ['OMR', 'Anna Nagar', 'Adyar', 'Velachery', 'T Nagar'],
            'Kolkata': ['Salt Lake', 'New Town', 'Ballygunge', 'Park Street', 'Howrah'],
            'Ahmedabad': ['SG Highway', 'Bopal', 'Thaltej', 'Vastrapur', 'Prahlad Nagar']
        }
        
        properties = []
        
        for prop_id in range(1, num_properties + 1):
            city = random.choice(list(cities.keys()))
            city_data = cities[city]
            
            # Determine property type based on distribution
            type_options = list(city_data['type_dist'].keys())
            type_weights = list(city_data['type_dist'].values())
            property_type = random.choices(type_options, weights=type_weights, k=1)[0]
            
            # Generate property details
            locality = random.choice(localities[city])
            bhk = random.choice([1, 2, 3, 4, 5])
            size_sqft = random.randint(500, 5000) if property_type in ['Flat', 'Apartment'] else random.randint(1000, 10000)
            
            # Calculate base price
            base_price = city_data['base_price']
            bhk_multiplier = {1: 0.7, 2: 1.0, 3: 1.3, 4: 1.6, 5: 2.0}[bhk]
            size_multiplier = size_sqft / 1000
            type_multiplier = {'Flat': 0.9, 'Apartment': 1.0, 'Villa': 2.0, 'Penthouse': 2.5, 'Independent House': 1.8}[property_type]
            
            price_lakhs = base_price * bhk_multiplier * size_multiplier * type_multiplier * random.uniform(0.8, 1.2)
            price_lakhs = round(price_lakhs, 1)
            
            # Generate other features
            age_years = random.randint(0, 30)
            furnished = random.choice(['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'])
            facing = random.choice(['North', 'South', 'East', 'West', 'North-East', 'North-West', 'South-East', 'South-West'])
            floor_no = random.randint(1, 30) if property_type in ['Flat', 'Apartment'] else 1
            total_floors = random.randint(5, 40) if property_type in ['Flat', 'Apartment'] else 3
            parking = random.randint(0, 3)
            
            # Amenities scores
            nearby_schools = random.randint(1, 10)
            nearby_hospitals = random.randint(1, 10)
            transport_access = random.randint(1, 10)
            amenities_score = random.randint(1, 10)
            
            # Calculate investment metrics
            location_score = random.uniform(6.0, 9.5)
            infrastructure_score = random.uniform(6.0, 9.0)
            growth_potential = city_data['growth'] * random.uniform(0.8, 1.2)
            
            # Calculate ROI
            base_roi = city_data['growth'] * 5
            property_specific_roi = base_roi * random.uniform(0.8, 1.3)
            
            # Determine investment status
            investment_score = (location_score * 0.3 + infrastructure_score * 0.2 + 
                               growth_potential * 0.3 + amenities_score * 0.1 + 
                               (10 - age_years/3) * 0.1)
            
            if investment_score >= 8:
                investment_status = 'Excellent Investment'
                status_color = '#00c853'
            elif investment_score >= 7:
                investment_status = 'Good Investment'
                status_color = '#00b0ff'
            elif investment_score >= 6:
                investment_status = 'Average Investment'
                status_color = '#ff9800'
            else:
                investment_status = 'High Risk'
                status_color = '#ff1744'
            
            properties.append({
                'Property ID': f'PROP{prop_id:04d}',
                'City': city,
                'Locality': locality,
                'Property Type': property_type,
                'BHK': bhk,
                'Size (sq ft)': size_sqft,
                'Price (₹ L)': price_lakhs,
                'Price per sq ft': round((price_lakhs * 100000) / size_sqft, 0),
                'Age (years)': age_years,
                'Furnished Status': furnished,
                'Facing': facing,
                'Floor': f'{floor_no}/{total_floors}',
                'Parking': parking,
                'Nearby Schools': nearby_schools,
                'Nearby Hospitals': nearby_hospitals,
                'Transport Access': transport_access,
                'Amenities Score': amenities_score,
                'Location Score': round(location_score, 1),
                'Infrastructure Score': round(infrastructure_score, 1),
                'Growth Potential (%)': round(growth_potential, 1),
                '5-Year ROI (%)': round(property_specific_roi, 1),
                'Investment Status': investment_status,
                'Status Color': status_color,
                'Investment Score': round(investment_score, 1),
                'Property Link': f'/property/{prop_id}',
                'Last Updated': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(properties)

# ============================================
# PROPERTY ANALYZER CLASS
# ============================================
class PropertyAnalyzer:
    """Advanced property analysis engine"""
    
    def __init__(self, properties_df):
        self.properties_df = properties_df
        self.market_insights = self.generate_market_insights()
    
    def generate_market_insights(self):
        """Generate comprehensive market insights"""
        insights = {}
        
        for city in self.properties_df['City'].unique():
            city_data = self.properties_df[self.properties_df['City'] == city]
            
            insights[city] = {
                'avg_price': city_data['Price (₹ L)'].mean(),
                'median_price': city_data['Price (₹ L)'].median(),
                'min_price': city_data['Price (₹ L)'].min(),
                'max_price': city_data['Price (₹ L)'].max(),
                'total_properties': len(city_data),
                'avg_roi': city_data['5-Year ROI (%)'].mean(),
                'top_localities': city_data.groupby('Locality').size().nlargest(5).to_dict(),
                'property_type_dist': city_data['Property Type'].value_counts().to_dict(),
                'bhk_distribution': city_data['BHK'].value_counts().to_dict(),
                'price_trend': 'Rising' if random.random() > 0.3 else 'Stable'
            }
        
        return insights
    
    def analyze_property(self, property_data):
        """Analyze a single property for investment potential"""
        
        # Get city insights
        city_insights = self.market_insights.get(property_data['City'], {})
        
        # Calculate price comparison
        price_ratio = property_data['Price (₹ L)'] / city_insights.get('avg_price', 1)
        price_status = 'Below Average' if price_ratio < 0.9 else 'Average' if price_ratio <= 1.1 else 'Above Average'
        
        # Calculate ROI potential
        base_roi = property_data['5-Year ROI (%)']
        city_avg_roi = city_insights.get('avg_roi', 8)
        roi_ratio = base_roi / city_avg_roi
        roi_status = 'High' if roi_ratio > 1.2 else 'Good' if roi_ratio > 1.0 else 'Average'
        
        # Calculate comprehensive score
        score_components = {
            'Price Value': max(0, min(10, 10 * (1 - abs(price_ratio - 1)))),
            'Location': property_data['Location Score'] / 10 * 10,
            'Amenities': property_data['Amenities Score'] / 10 * 10,
            'Infrastructure': property_data['Infrastructure Score'] / 10 * 10,
            'Growth Potential': property_data['Growth Potential (%)'] / 10 * 10,
            'Property Condition': max(0, 10 - property_data['Age (years)'] * 0.3)
        }
        
        total_score = sum(score_components.values()) / len(score_components)
        
        # Determine recommendation
        if total_score >= 8:
            recommendation = 'STRONG BUY'
            recommendation_color = '#00c853'
            confidence = random.uniform(85, 95)
        elif total_score >= 7:
            recommendation = 'BUY'
            recommendation_color = '#00b0ff'
            confidence = random.uniform(70, 85)
        elif total_score >= 6:
            recommendation = 'HOLD'
            recommendation_color = '#ff9800'
            confidence = random.uniform(60, 70)
        else:
            recommendation = 'AVOID'
            recommendation_color = '#ff1744'
            confidence = random.uniform(40, 60)
        
        # Generate insights
        insights = [
            f"Property is {price_status} priced compared to city average",
            f"Expected 5-year ROI: {base_roi}% ({roi_status} return)",
            f"Location score of {property_data['Location Score']}/10 indicates good connectivity",
            f"Property age: {property_data['Age (years)']} years - {'New' if property_data['Age (years)'] <= 5 else 'Established'} property",
            f"Amenities score: {property_data['Amenities Score']}/10 - {'Good' if property_data['Amenities Score'] >= 7 else 'Average'} facilities"
        ]
        
        # Calculate future value
        future_value = property_data['Price (₹ L)'] * ((1 + base_roi/100) ** 5)
        
        return {
            'total_score': round(total_score, 1),
            'score_components': score_components,
            'recommendation': recommendation,
            'recommendation_color': recommendation_color,
            'confidence': round(confidence, 1),
            'price_status': price_status,
            'roi_status': roi_status,
            'insights': insights,
            'future_value': round(future_value, 1),
            'annual_appreciation': round(base_roi / 5, 1)
        }

# ============================================
# MAIN APPLICATION
# ============================================
class RealEstateAdvisorUltimate:
    def __init__(self):
        # Initialize session state
        if 'properties_df' not in st.session_state:
            st.session_state.properties_df = PropertyDataGenerator.generate_property_data(200)
        
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = PropertyAnalyzer(st.session_state.properties_df)
        
        if 'current_property' not in st.session_state:
            st.session_state.current_property = None
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        self.properties_df = st.session_state.properties_df
        self.analyzer = st.session_state.analyzer
    
    def show_header(self):
        """Show premium header"""
        st.markdown('<h1 class="premium-header">🏢 REAL ESTATE PRO ADVISOR</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header-pro">AI-Powered Property Intelligence Platform | Smart Investment Decisions</p>', unsafe_allow_html=True)
        
        # Quick stats ribbon
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🏙️ Cities", "8", "Covered")
        with col2:
            st.metric("🏘️ Properties", f"{len(self.properties_df):,}", "Available")
        with col3:
            avg_price = self.properties_df['Price (₹ L)'].mean()
            st.metric("💰 Avg Price", f"₹{avg_price:,.0f}L", "Market Avg")
        with col4:
            avg_roi = self.properties_df['5-Year ROI (%)'].mean()
            st.metric("📈 Avg ROI", f"{avg_roi:.1f}%", "Annual")
        with col5:
            good_investments = len(self.properties_df[self.properties_df['Investment Status'].str.contains('Investment')])
            st.metric("🎯 Good Deals", f"{good_investments:,}", "Found")
    
    def show_search_sidebar(self):
        """Show advanced search sidebar"""
        with st.sidebar:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1a237e 0%, #283593 100%); 
                       padding: 25px; border-radius: 20px; margin-bottom: 25px;'>
                <h3 style='color: white; margin: 0 0 10px 0;'>🔍 ADVANCED SEARCH</h3>
                <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>
                    Find your perfect property with intelligent filters
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("property_search"):
                # Location Filters
                st.markdown("### 📍 Location")
                selected_cities = st.multiselect(
                    "Select Cities",
                    options=sorted(self.properties_df['City'].unique()),
                    default=['Mumbai', 'Bangalore', 'Delhi']
                )
                
                # Property Type with emphasis on Flats
                st.markdown("### 🏠 Property Type")
                property_types = st.multiselect(
                    "Select Types",
                    options=sorted(self.properties_df['Property Type'].unique()),
                    default=['Flat', 'Apartment']
                )
                
                # BHK Configuration
                st.markdown("### 🛏️ Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    min_bhk = st.selectbox("Min BHK", [1, 2, 3, 4, 5], index=1)
                with col2:
                    max_bhk = st.selectbox("Max BHK", [1, 2, 3, 4, 5], index=3)
                
                # Price Range
                st.markdown("### 💰 Price Range (₹ Lakhs)")
                min_price, max_price = st.slider(
                    "Select Price Range",
                    min_value=int(self.properties_df['Price (₹ L)'].min()),
                    max_value=int(self.properties_df['Price (₹ L)'].max()),
                    value=(50, 500),
                    step=10,
                    label_visibility="collapsed"
                )
                
                # Size Range
                st.markdown("### 📐 Size Range (sq ft)")
                min_size, max_size = st.slider(
                    "Select Size Range",
                    min_value=int(self.properties_df['Size (sq ft)'].min()),
                    max_value=int(self.properties_df['Size (sq ft)'].max()),
                    value=(500, 3000),
                    step=100,
                    label_visibility="collapsed"
                )
                
                # Advanced Filters
                with st.expander("⚙️ Advanced Filters", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        furnished_status = st.multiselect(
                            "Furnishing",
                            options=sorted(self.properties_df['Furnished Status'].unique()),
                            default=sorted(self.properties_df['Furnished Status'].unique())
                        )
                    
                    with col2:
                        min_roi = st.slider("Min ROI (%)", 5.0, 20.0, 8.0, 0.5)
                
                # Search button
                search_button = st.form_submit_button(
                    "🚀 SEARCH PROPERTIES",
                    use_container_width=True
                )
                
                if search_button:
                    # Apply filters
                    filtered_df = self.properties_df[
                        (self.properties_df['City'].isin(selected_cities)) &
                        (self.properties_df['Property Type'].isin(property_types)) &
                        (self.properties_df['BHK'] >= min_bhk) &
                        (self.properties_df['BHK'] <= max_bhk) &
                        (self.properties_df['Price (₹ L)'] >= min_price) &
                        (self.properties_df['Price (₹ L)'] <= max_price) &
                        (self.properties_df['Size (sq ft)'] >= min_size) &
                        (self.properties_df['Size (sq ft)'] <= max_size) &
                        (self.properties_df['Furnished Status'].isin(furnished_status)) &
                        (self.properties_df['5-Year ROI (%)'] >= min_roi)
                    ]
                    
                    st.session_state.filtered_properties = filtered_df
                    st.session_state.show_results = True
                    st.session_state.current_view = 'search_results'
                    st.rerun()
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Dashboard", use_container_width=True):
                    st.session_state.current_view = 'dashboard'
                    st.rerun()
            
            with col2:
                if st.button("🤖 AI Analysis", use_container_width=True):
                    st.session_state.current_view = 'ai_analysis'
                    st.rerun()
            
            # Market Insights
            st.markdown("---")
            st.markdown("### 📈 Market Insights")
            
            if st.session_state.get('current_property'):
                prop = st.session_state.current_property
                city_insights = self.analyzer.market_insights.get(prop['City'], {})
                
                st.metric(
                    f"{prop['City']} Avg Price",
                    f"₹{city_insights.get('avg_price', 0):,.0f}L"
                )
                st.metric(
                    "City Growth",
                    f"{city_insights.get('avg_roi', 0):.1f}%",
                    "5-Year ROI"
                )
    
    def show_property_cards(self, properties_df):
        """Display property cards in grid view"""
        if len(properties_df) == 0:
            st.warning("No properties found matching your criteria. Try adjusting your filters.")
            return
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Found", f"{len(properties_df)}", "Properties")
        with col2:
            avg_price = properties_df['Price (₹ L)'].mean()
            st.metric("Avg Price", f"₹{avg_price:,.0f}L")
        with col3:
            avg_roi = properties_df['5-Year ROI (%)'].mean()
            st.metric("Avg ROI", f"{avg_roi:.1f}%")
        with col4:
            good_deals = len(properties_df[properties_df['Investment Status'].str.contains('Investment')])
            st.metric("Good Deals", good_deals)
        
        # Sort options
        sort_options = {
            'Price: Low to High': 'Price (₹ L)',
            'Price: High to Low': 'Price (₹ L)',
            'ROI: High to Low': '5-Year ROI (%)',
            'Investment Score': 'Investment Score',
            'Size: Large First': 'Size (sq ft)'
        }
        
        selected_sort = st.selectbox("Sort By", list(sort_options.keys()))
        sort_column = sort_options[selected_sort]
        ascending = 'Low to High' in selected_sort
        
        sorted_df = properties_df.sort_values(by=sort_column, ascending=ascending)
        
        # Display properties in grid
        cols_per_row = 3
        num_properties = len(sorted_df)
        
        for i in range(0, num_properties, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_properties:
                    with cols[j]:
                        self.show_property_card(sorted_df.iloc[i + j])
    
    def show_property_card(self, property_data):
        """Display individual property card"""
        
        status_color = property_data['Status Color']
        
        st.markdown(f"""
        <div class="property-card-premium">
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;'>
                <div>
                    <h4 style='margin: 0; color: #1a237e;'>{property_data['Property ID']}</h4>
                    <p style='margin: 5px 0; color: #546e7a; font-size: 0.9rem;'>
                        📍 {property_data['City']} • {property_data['Locality']}
                    </p>
                </div>
                <span style='background: {status_color}; color: white; padding: 6px 15px; 
                      border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;'>
                    {property_data['Investment Status'].upper()}
                </span>
            </div>
            
            <div style='background: #f5f7ff; padding: 15px; border-radius: 15px; margin: 15px 0;'>
                <div style='text-align: center;'>
                    <div style='font-size: 1.8rem; font-weight: 800; color: #1a237e;'>
                        ₹{property_data['Price (₹ L)']:,.1f}L
                    </div>
                    <div style='font-size: 0.9rem; color: #546e7a;'>
                        ₹{property_data['Price per sq ft']:,.0f}/sq ft
                    </div>
                </div>
            </div>
            
            <div style='margin: 20px 0;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 0.8rem; color: #78909c;'>TYPE</div>
                        <div style='font-weight: 600; color: #37474f;'>{property_data['Property Type']}</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 0.8rem; color: #78909c;'>BHK</div>
                        <div style='font-weight: 600; color: #37474f;'>{property_data['BHK']}</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 0.8rem; color: #78909c;'>SIZE</div>
                        <div style='font-weight: 600; color: #37474f;'>{property_data['Size (sq ft)']} sq ft</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 0.8rem; color: #78909c;'>ROI</div>
                        <div style='font-weight: 600; color: #2e7d32;'>{property_data['5-Year ROI (%)']}%</div>
                    </div>
                </div>
            </div>
            
            <div style='background: linear-gradient(90deg, #e8eaf6 0%, #c5cae9 100%); 
                  padding: 12px; border-radius: 12px; margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'>
                    <div style='text-align: center; flex: 1;'>
                        <div style='color: #3949ab; font-weight: 600;'>📍 {property_data['Location Score']}/10</div>
                        <div style='font-size: 0.7rem; color: #5c6bc0;'>Location</div>
                    </div>
                    <div style='text-align: center; flex: 1; border-left: 1px solid #9fa8da; border-right: 1px solid #9fa8da;'>
                        <div style='color: #3949ab; font-weight: 600;'>🏥 {property_data['Amenities Score']}/10</div>
                        <div style='font-size: 0.7rem; color: #5c6bc0;'>Amenities</div>
                    </div>
                    <div style='text-align: center; flex: 1;'>
                        <div style='color: #3949ab; font-weight: 600;'>📈 {property_data['Investment Score']}/10</div>
                        <div style='font-size: 0.7rem; color: #5c6bc0;'>Score</div>
                    </div>
                </div>
            </div>
            
            <button onclick="analyzeProperty('{property_data['Property ID']}')" 
                    style='width: 100%; background: linear-gradient(135deg, #3949ab 0%, #283593 100%); 
                           color: white; border: none; padding: 12px; border-radius: 12px; 
                           font-weight: 600; cursor: pointer; margin-top: 10px;'>
                🔍 Analyze Property
            </button>
        </div>
        """, unsafe_allow_html=True)
    
    def show_property_analysis(self, property_data):
        """Show detailed property analysis"""
        
        # Analyze property
        analysis = self.analyzer.analyze_property(property_data)
        
        st.markdown(f"""
        <div class="premium-card">
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;'>
                <div>
                    <h2 style='margin: 0; color: #1a237e;'>Property Analysis: {property_data['Property ID']}</h2>
                    <p style='margin: 5px 0 0 0; color: #546e7a;'>
                        📍 {property_data['City']} • {property_data['Locality']} • {property_data['Property Type']}
                    </p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 2.5rem; font-weight: 800; color: #1a237e;'>
                        {analysis['total_score']}/10
                    </div>
                    <div style='font-size: 0.9rem; color: #546e7a;'>Overall Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation Banner
        if analysis['recommendation'] == 'STRONG BUY':
            st.markdown(f"""
            <div class="investment-premium-good">
                {analysis['recommendation']} • Confidence: {analysis['confidence']}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="investment-premium-bad">
                {analysis['recommendation']} • Confidence: {analysis['confidence']}%
            </div>
            """, unsafe_allow_html=True)
        
        # Price Forecast
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Price Analysis")
            
            st.metric(
                "Current Price",
                f"₹{property_data['Price (₹ L)']:,.1f}L",
                delta=f"{analysis['price_status']}"
            )
            
            st.metric(
                "Future Value (5 Years)",
                f"₹{analysis['future_value']:,.1f}L",
                delta=f"+{analysis['annual_appreciation']}% p.a."
            )
            
            st.metric(
                "5-Year ROI",
                f"{property_data['5-Year ROI (%)']}%",
                delta=f"{analysis['roi_status']} Return"
            )
            
            # Price projection chart
            years = list(range(6))
            prices = [
                property_data['Price (₹ L)'],
                property_data['Price (₹ L)'] * (1 + analysis['annual_appreciation']/100),
                property_data['Price (₹ L)'] * ((1 + analysis['annual_appreciation']/100) ** 2),
                property_data['Price (₹ L)'] * ((1 + analysis['annual_appreciation']/100) ** 3),
                property_data['Price (₹ L)'] * ((1 + analysis['annual_appreciation']/100) ** 4),
                analysis['future_value']
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=prices,
                mode='lines+markers',
                name='Price Projection',
                line=dict(color='#00c853', width=4),
                marker=dict(size=10, color='#00c853'),
                fill='tozeroy',
                fillcolor='rgba(0, 200, 83, 0.1)'
            ))
            
            fig.update_layout(
                title="📈 5-Year Price Projection",
                xaxis_title="Years",
                yaxis_title="Price (₹ Lakhs)",
                height=300,
                template="plotly_white",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Investment Score Breakdown")
            
            # Score components visualization
            components_df = pd.DataFrame({
                'Component': list(analysis['score_components'].keys()),
                'Score': list(analysis['score_components'].values())
            })
            
            fig = px.bar(
                components_df,
                y='Component',
                x='Score',
                orientation='h',
                color='Score',
                color_continuous_scale='RdYlGn',
                range_color=[0, 10],
                labels={'Score': 'Score (0-10)'}
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_range=[0, 10]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            st.markdown("### 🎯 Key Metrics")
            
            metrics = [
                ("Location Score", f"{property_data['Location Score']}/10", "#2196f3"),
                ("Amenities Score", f"{property_data['Amenities Score']}/10", "#4caf50"),
                ("Infrastructure Score", f"{property_data['Infrastructure Score']}/10", "#ff9800"),
                ("Growth Potential", f"{property_data['Growth Potential (%)']}%", "#9c27b0"),
                ("Property Age", f"{property_data['Age (years)']} years", "#607d8b")
            ]
            
            for name, value, color in metrics:
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; 
                          padding: 12px; background: #f5f7ff; border-radius: 10px; margin: 8px 0;'>
                    <span style='font-weight: 600; color: #37474f;'>{name}</span>
                    <span style='font-weight: 700; color: {color};'>{value}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Property Details
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Property Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏗️ Specifications")
            details = [
                ("Property Type", property_data['Property Type']),
                ("BHK", property_data['BHK']),
                ("Size", f"{property_data['Size (sq ft)']} sq ft"),
                ("Price per sq ft", f"₹{property_data['Price per sq ft']:,.0f}")
            ]
            
            for label, value in details:
                st.markdown(f"**{label}:** {value}")
        
        with col2:
            st.markdown("#### 🎯 Features")
            features = [
                ("Furnishing", property_data['Furnished Status']),
                ("Facing", property_data['Facing']),
                ("Floor", property_data['Floor']),
                ("Parking", f"{property_data['Parking']} spots")
            ]
            
            for label, value in features:
                st.markdown(f"**{label}:** {value}")
        
        with col3:
            st.markdown("#### 🏥 Amenities")
            amenities = [
                ("Schools", f"{property_data['Nearby Schools']}/10"),
                ("Hospitals", f"{property_data['Nearby Hospitals']}/10"),
                ("Transport", f"{property_data['Transport Access']}/10"),
                ("Last Updated", property_data['Last Updated'])
            ]
            
            for label, value in amenities:
                st.markdown(f"**{label}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Market Comparison
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 🏙️ Market Comparison")
        
        city_insights = self.analyzer.market_insights.get(property_data['City'], {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "City Avg Price",
                f"₹{city_insights.get('avg_price', 0):,.0f}L",
                delta=f"{'Below' if property_data['Price (₹ L)'] < city_insights.get('avg_price', 0) else 'Above'} Avg"
            )
        
        with col2:
            st.metric(
                "City Avg ROI",
                f"{city_insights.get('avg_roi', 0):.1f}%",
                delta=f"{'Higher' if property_data['5-Year ROI (%)'] > city_insights.get('avg_roi', 0) else 'Lower'}"
            )
        
        with col3:
            st.metric(
                "Similar Properties",
                city_insights.get('total_properties', 0),
                "In City"
            )
        
        with col4:
            st.metric(
                "Market Trend",
                city_insights.get('price_trend', 'Stable'),
                "Outlook"
            )
        
        # Similar properties comparison
        similar_props = self.properties_df[
            (self.properties_df['City'] == property_data['City']) &
            (self.properties_df['Property Type'] == property_data['Property Type']) &
            (self.properties_df['BHK'] == property_data['BHK']) &
            (self.properties_df['Property ID'] != property_data['Property ID'])
        ].head(5)
        
        if len(similar_props) > 0:
            st.markdown("#### 📊 Similar Properties in Area")
            
            comparison_data = []
            for _, prop in similar_props.iterrows():
                comparison_data.append({
                    'Property': prop['Property ID'],
                    'Price (₹ L)': prop['Price (₹ L)'],
                    'ROI (%)': prop['5-Year ROI (%)'],
                    'Score': prop['Investment Score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.background_gradient(subset=['ROI (%)', 'Score'], cmap='RdYlGn'), 
                        use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Recommendations
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Action Plan & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Next Steps")
            
            if analysis['recommendation'] in ['STRONG BUY', 'BUY']:
                steps = [
                    "📞 Schedule property visit with agent",
                    "📝 Verify all legal documents",
                    "💰 Arrange financing options",
                    "🔍 Get professional property inspection",
                    "⚖️ Consult with real estate lawyer"
                ]
            else:
                steps = [
                    "💰 Negotiate for better price",
                    "🔍 Explore alternative properties",
                    "📊 Wait for market correction",
                    "🏙️ Consider different locality",
                    "👨‍💼 Consult investment advisor"
                ]
            
            for step in steps:
                st.markdown(f"- {step}")
        
        with col2:
            st.markdown("#### 📈 Investment Strategy")
            
            strategies = [
                f"**Hold Period:** {random.choice(['5-7 years', '7-10 years', 'Long-term'])} recommended",
                f"**Exit Strategy:** {random.choice(['Sell after appreciation', 'Rent out', 'Hold for redevelopment'])}",
                f"**Risk Level:** {random.choice(['Low', 'Medium', 'Moderate'])}",
                f"**Expected Annual Return:** {analysis['annual_appreciation']}%",
                f"**Breakeven Period:** {random.choice(['4-5 years', '5-6 years', '6-7 years'])}"
            ]
            
            for strategy in strategies:
                st.markdown(f"- {strategy}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_dashboard(self):
        """Show comprehensive dashboard"""
        st.markdown("## 📊 MARKET INTELLIGENCE DASHBOARD")
        
        # Top row: Key insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Top Performing Cities")
            
            city_performance = []
            for city, insights in self.analyzer.market_insights.items():
                city_performance.append({
                    'City': city,
                    'Avg ROI': insights['avg_roi'],
                    'Avg Price': insights['avg_price']
                })
            
            perf_df = pd.DataFrame(city_performance).sort_values('Avg ROI', ascending=False).head(5)
            
            for _, row in perf_df.iterrows():
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; 
                          padding: 12px; background: #f5f7ff; border-radius: 10px; margin: 8px 0;'>
                    <span style='font-weight: 600;'>{row['City']}</span>
                    <span style='font-weight: 700; color: #2e7d32;'>{row['Avg ROI']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Market Trends")
            
            # Price trend chart
            cities = list(self.analyzer.market_insights.keys())[:5]
            prices = [self.analyzer.market_insights[c]['avg_price'] for c in cities]
            growth = [self.analyzer.market_insights[c]['avg_roi'] for c in cities]
            
            fig = go.Figure(data=[
                go.Bar(name='Avg Price', x=cities, y=prices, marker_color='#3949ab'),
                go.Bar(name='Avg ROI', x=cities, y=growth, marker_color='#00c853')
            ])
            
            fig.update_layout(
                height=300,
                barmode='group',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Investment Opportunities")
            
            # Find best opportunities
            opportunities = self.properties_df[
                (self.properties_df['Investment Score'] >= 7.5) &
                (self.properties_df['5-Year ROI (%)'] >= 10)
            ].sort_values('Investment Score', ascending=False).head(5)
            
            if len(opportunities) > 0:
                for _, opp in opportunities.iterrows():
                    st.markdown(f"""
                    <div style='padding: 12px; background: #e8f5e9; border-radius: 10px; margin: 8px 0; 
                              border-left: 4px solid #00c853;'>
                        <div style='font-weight: 600;'>{opp['Property ID']}</div>
                        <div style='font-size: 0.9rem; color: #546e7a;'>
                            {opp['City']} • {opp['Property Type']} • ₹{opp['Price (₹ L)']:,.0f}L
                        </div>
                        <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                            <span style='color: #2e7d32; font-weight: 600;'>{opp['5-Year ROI (%)']}% ROI</span>
                            <span style='color: #3949ab; font-weight: 600;'>{opp['Investment Score']}/10</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-value opportunities found in current market")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Market analysis charts
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Comprehensive Market Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🏙️ City Comparison", "📈 ROI Distribution", "💰 Price Analysis"])
        
        with tab1:
            # City comparison heatmap
            cities = list(self.analyzer.market_insights.keys())
            metrics = ['avg_price', 'avg_roi', 'total_properties']
            
            heatmap_data = []
            for city in cities:
                row = [self.analyzer.market_insights[city][metric] for metric in metrics]
                heatmap_data.append(row)
            
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Metric", y="City", color="Value"),
                x=['Avg Price', 'Avg ROI', 'Properties'],
                y=cities,
                color_continuous_scale='Viridis',
                aspect="auto"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # ROI distribution
            fig = px.histogram(
                self.properties_df,
                x='5-Year ROI (%)',
                nbins=20,
                title='ROI Distribution Across Properties',
                labels={'5-Year ROI (%)': '5-Year ROI (%)'},
                color_discrete_sequence=['#00c853']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Price analysis
            fig = px.scatter(
                self.properties_df.sample(min(100, len(self.properties_df))),
                x='Size (sq ft)',
                y='Price (₹ L)',
                color='5-Year ROI (%)',
                size='Investment Score',
                hover_name='Property ID',
                title='Price vs Size Analysis',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        self.show_header()
        self.show_search_sidebar()
        
        # Initialize session state
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'dashboard'
        
        if 'show_results' not in st.session_state:
            st.session_state.show_results = False
        
        # Show the selected view
        if st.session_state.current_view == 'dashboard':
            self.show_dashboard()
        elif st.session_state.current_view == 'search_results' and st.session_state.show_results:
            filtered_df = st.session_state.get('filtered_properties', self.properties_df)
            self.show_property_cards(filtered_df)
        elif st.session_state.current_view == 'ai_analysis':
            self.show_property_analysis(st.session_state.current_property) \
                if st.session_state.current_property else self.show_dashboard()
        
        # Add JavaScript for property selection
        st.markdown("""
        <script>
        function analyzeProperty(propertyId) {
            // This would normally send a request to the backend
            alert('Analyzing property: ' + propertyId);
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="footer-premium">
            <div style='text-align: center;'>
                <h3 style='color: white; margin-bottom: 20px;'>🏢 RealEstate Pro Advisor</h3>
                <p style='color: rgba(255,255,255,0.8); margin-bottom: 25px;'>
                    AI-Powered Real Estate Intelligence Platform • Making Smart Investment Decisions Easy
                </p>
                <div style='display: flex; justify-content: center; gap: 20px;'>
                    <span style='background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 20px; 
                          color: white; font-size: 0.9rem;'>🤖 Machine Learning</span>
                    <span style='background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 20px; 
                          color: white; font-size: 0.9rem;'>📊 Data Analytics</span>
                    <span style='background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 20px; 
                          color: white; font-size: 0.9rem;'>🏢 Real Estate</span>
                    <span style='background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 20px; 
                          color: white; font-size: 0.9rem;'>📈 Investment</span>
                </div>
                <p style='color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 30px;'>
                    © 2024 RealEstate Pro Advisor • Built with Streamlit • Data for demonstration purposes only
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    app = RealEstateAdvisorUltimate()
    app.run()
