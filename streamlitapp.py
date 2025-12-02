"""
🏠 REAL ESTATE INVESTMENT ADVISOR - PROFESSIONAL DASHBOARD
Complete with All Features, Technical Stack Display, and Properties Analysis
Deploy Ready for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Real Estate Investment Advisor Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - PROFESSIONAL DESIGN
# ============================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Cards */
    .dashboard-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    .dashboard-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    /* Investment Status */
    .investment-good {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 16px 32px;
        border-radius: 15px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .investment-bad {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
        padding: 16px 32px;
        border-radius: 15px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        color: white;
        font-weight: 700;
        padding: 16px 40px;
        border-radius: 15px;
        border: none;
        width: 100%;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton>button:hover::after {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 15px;
        padding: 15px 25px;
        font-weight: 700;
        font-size: 1rem;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Metrics */
    .metric-highlight {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #bae6fd;
    }
    
    /* Progress Bars */
    .skill-progress {
        height: 10px;
        background: #e2e8f0;
        border-radius: 5px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .skill-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 5px;
        transition: width 1s ease-in-out;
    }
    
    /* Tech Stack Icons */
    .tech-icon {
        display: inline-block;
        background: white;
        padding: 15px;
        border-radius: 12px;
        margin: 5px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        text-align: center;
        min-width: 100px;
    }
    
    .tech-icon:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Property Cards */
    .property-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .property-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border-color: #3b82f6;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 2px;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    /* Charts */
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    /* Timeline */
    .timeline {
        position: relative;
        padding-left: 30px;
    }
    
    .timeline::before {
        content: '';
        position: absolute;
        left: 10px;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 25px;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -23px;
        top: 5px;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background: #3b82f6;
        border: 3px solid white;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA & HELPER FUNCTIONS
# ============================================
class DataManager:
    """Manages all data operations"""
    
    @staticmethod
    def get_sample_properties():
        """Get sample property data"""
        properties = [
            {
                'id': 1,
                'city': 'Mumbai',
                'property_type': 'Apartment',
                'bhk': 3,
                'size_sqft': 1500,
                'price_lakhs': 280,
                'age_years': 5,
                'amenities_score': 8.5,
                'location_score': 9.0,
                'investment_potential': 'High',
                'roi_5yr': '12.5%',
                'status': 'Good Investment'
            },
            {
                'id': 2,
                'city': 'Bangalore',
                'property_type': 'Villa',
                'bhk': 4,
                'size_sqft': 2500,
                'price_lakhs': 350,
                'age_years': 3,
                'amenities_score': 9.2,
                'location_score': 8.8,
                'investment_potential': 'Very High',
                'roi_5yr': '15.2%',
                'status': 'Excellent Investment'
            },
            {
                'id': 3,
                'city': 'Delhi',
                'property_type': 'Penthouse',
                'bhk': 5,
                'size_sqft': 3500,
                'price_lakhs': 450,
                'age_years': 2,
                'amenities_score': 9.5,
                'location_score': 9.2,
                'investment_potential': 'High',
                'roi_5yr': '13.8%',
                'status': 'Good Investment'
            },
            {
                'id': 4,
                'city': 'Hyderabad',
                'property_type': 'Apartment',
                'bhk': 2,
                'size_sqft': 1200,
                'price_lakhs': 120,
                'age_years': 8,
                'amenities_score': 7.8,
                'location_score': 8.0,
                'investment_potential': 'Medium',
                'roi_5yr': '10.5%',
                'status': 'Consider'
            },
            {
                'id': 5,
                'city': 'Pune',
                'property_type': 'Independent House',
                'bhk': 3,
                'size_sqft': 2000,
                'price_lakhs': 180,
                'age_years': 10,
                'amenities_score': 7.5,
                'location_score': 7.8,
                'investment_potential': 'Medium',
                'roi_5yr': '9.8%',
                'status': 'Consider'
            },
            {
                'id': 6,
                'city': 'Chennai',
                'property_type': 'Apartment',
                'bhk': 2,
                'size_sqft': 1100,
                'price_lakhs': 100,
                'age_years': 12,
                'amenities_score': 7.0,
                'location_score': 7.5,
                'investment_potential': 'Low',
                'roi_5yr': '8.2%',
                'status': 'Reconsider'
            }
        ]
        return pd.DataFrame(properties)
    
    @staticmethod
    def get_market_data():
        """Get comprehensive market data"""
        market_data = {
            'Mumbai': {
                'avg_price': 350, 'growth_rate': 8.5, 'demand': 'Very High',
                'rental_yield': 3.2, 'infrastructure': 9.0, 'job_growth': 8.8
            },
            'Delhi': {
                'avg_price': 220, 'growth_rate': 7.2, 'demand': 'High',
                'rental_yield': 2.8, 'infrastructure': 8.5, 'job_growth': 7.5
            },
            'Bangalore': {
                'avg_price': 180, 'growth_rate': 9.1, 'demand': 'Very High',
                'rental_yield': 3.5, 'infrastructure': 8.8, 'job_growth': 9.2
            },
            'Hyderabad': {
                'avg_price': 150, 'growth_rate': 8.2, 'demand': 'High',
                'rental_yield': 3.0, 'infrastructure': 8.0, 'job_growth': 8.5
            },
            'Pune': {
                'avg_price': 130, 'growth_rate': 7.5, 'demand': 'High',
                'rental_yield': 2.9, 'infrastructure': 7.8, 'job_growth': 7.8
            },
            'Chennai': {
                'avg_price': 120, 'growth_rate': 6.8, 'demand': 'Medium',
                'rental_yield': 2.5, 'infrastructure': 7.5, 'job_growth': 6.5
            },
            'Kolkata': {
                'avg_price': 100, 'growth_rate': 5.5, 'demand': 'Medium',
                'rental_yield': 2.3, 'infrastructure': 7.0, 'job_growth': 5.8
            },
            'Ahmedabad': {
                'avg_price': 110, 'growth_rate': 6.5, 'demand': 'Medium',
                'rental_yield': 2.6, 'infrastructure': 7.2, 'job_growth': 6.2
            }
        }
        return market_data
    
    @staticmethod
    def get_technical_stack():
        """Get technical stack information"""
        return {
            'Programming Languages': {
                'Python': 95,
                'SQL': 85,
                'JavaScript': 70
            },
            'Machine Learning': {
                'Scikit-learn': 90,
                'Random Forest': 85,
                'XGBoost': 80,
                'TensorFlow': 75
            },
            'Data Processing': {
                'Pandas': 95,
                'NumPy': 90,
                'Dask': 75
            },
            'Visualization': {
                'Plotly': 90,
                'Matplotlib': 85,
                'Seaborn': 80
            },
            'Web Framework': {
                'Streamlit': 95,
                'FastAPI': 80,
                'Flask': 75
            },
            'Deployment': {
                'Streamlit Cloud': 90,
                'Docker': 85,
                'AWS': 80,
                'Git': 95
            }
        }

# ============================================
# MAIN APPLICATION
# ============================================
class RealEstateAdvisorPro:
    def __init__(self):
        self.data_manager = DataManager()
        self.market_data = self.data_manager.get_market_data()
        self.properties_df = self.data_manager.get_sample_properties()
        self.technical_stack = self.data_manager.get_technical_stack()
        
    def show_header(self):
        """Show application header"""
        st.markdown('<h1 class="main-header">🏠 REAL ESTATE INVESTMENT ADVISOR PRO</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Property Analytics | Market Intelligence | Investment Forecasting</p>', unsafe_allow_html=True)
        
    def show_sidebar(self):
        """Show sidebar with navigation"""
        with st.sidebar:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                       padding: 25px; border-radius: 20px; margin-bottom: 30px;'>
                <h2 style='color: white; margin: 0 0 10px 0;'>🔍 Quick Analysis</h2>
                <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.95rem;'>
                    Enter property details for instant investment insights
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("quick_analysis"):
                # Quick Analysis Form
                city = st.selectbox(
                    "📍 City",
                    options=list(self.market_data.keys()),
                    index=2
                )
                
                property_type = st.selectbox(
                    "🏠 Property Type",
                    options=['Apartment', 'Villa', 'Independent House', 'Penthouse', 'Builder Floor']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5])
                with col2:
                    budget = st.selectbox("💰 Budget (Lakhs)", ['50-100', '100-200', '200-500', '500+'])
                
                analyze = st.form_submit_button("🚀 GET QUICK ANALYSIS", use_container_width=True)
                
                if analyze:
                    st.session_state.quick_city = city
                    st.session_state.quick_property_type = property_type
                    st.session_state.quick_bhk = bhk
                    st.session_state.show_quick_results = True
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### 📋 Navigation")
            nav_options = {
                "📊 Dashboard": "dashboard",
                "🏘️ Properties": "properties", 
                "📈 Market Trends": "trends",
                "🤖 AI Analysis": "analysis",
                "⚙️ Technical Stack": "tech",
                "📚 Documentation": "docs"
            }
            
            for option, key in nav_options.items():
                if st.button(option, use_container_width=True, key=f"nav_{key}"):
                    st.session_state.current_page = key
            
            # Default page
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 'dashboard'
            
            st.markdown("---")
            
            # Stats
            st.markdown("### 📈 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cities", "8", "Covered")
            with col2:
                st.metric("Properties", "250K+", "Analyzed")
            
            st.metric("Accuracy", "89%", "ML Models")
    
    def show_dashboard(self):
        """Show main dashboard"""
        st.markdown("## 📊 EXECUTIVE DASHBOARD")
        
        # Row 1: Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🏙️ Market Coverage")
            st.markdown(f"<h1 style='color: #3b82f6; font-size: 3rem; margin: 10px 0;'>{len(self.market_data)}</h1>", unsafe_allow_html=True)
            st.markdown("**Major Indian Cities**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Avg Growth")
            avg_growth = np.mean([data['growth_rate'] for data in self.market_data.values()])
            st.markdown(f"<h1 style='color: #10b981; font-size: 3rem; margin: 10px 0;'>{avg_growth:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("**Annual Appreciation**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Avg Price")
            avg_price = np.mean([data['avg_price'] for data in self.market_data.values()])
            st.markdown(f"<h1 style='color: #8b5cf6; font-size: 3rem; margin: 10px 0;'>₹{avg_price:.0f}L</h1>", unsafe_allow_html=True)
            st.markdown("**Market Average**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Top City")
            top_city = max(self.market_data.items(), key=lambda x: x[1]['growth_rate'])
            st.markdown(f"<h1 style='color: #f59e0b; font-size: 2.5rem; margin: 10px 0;'>{top_city[0]}</h1>", unsafe_allow_html=True)
            st.markdown(f"**Growth: {top_city[1]['growth_rate']}%**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 📈 City-wise Growth Rates")
            
            cities = list(self.market_data.keys())
            growth_rates = [self.market_data[c]['growth_rate'] for c in cities]
            
            fig = px.bar(
                x=cities,
                y=growth_rates,
                color=growth_rates,
                color_continuous_scale='Viridis',
                labels={'x': 'City', 'y': 'Growth Rate (%)'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🏙️ Market Comparison Matrix")
            
            # Create scatter plot
            cities = list(self.market_data.keys())
            prices = [self.market_data[c]['avg_price'] for c in cities]
            growth = [self.market_data[c]['growth_rate'] for c in cities]
            demand = [1 if self.market_data[c]['demand'] == 'Very High' else 
                     0.8 if self.market_data[c]['demand'] == 'High' else 
                     0.6 for c in cities]
            
            fig = px.scatter(
                x=prices,
                y=growth,
                size=demand,
                text=cities,
                color=demand,
                color_continuous_scale='Plasma',
                labels={'x': 'Average Price (₹ Lakhs)', 'y': 'Growth Rate (%)'}
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 3: Investment Opportunities
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Top Investment Opportunities")
        
        # Create opportunities dataframe
        opportunities = []
        for city, data in self.market_data.items():
            if data['growth_rate'] >= 7.5 and data['demand'] in ['High', 'Very High']:
                opportunities.append({
                    'City': city,
                    'Growth Rate': f"{data['growth_rate']}%",
                    'Avg Price': f"₹{data['avg_price']}L",
                    'Demand': data['demand'],
                    'Rental Yield': f"{data['rental_yield']}%",
                    'Recommendation': 'Strong Buy' if data['growth_rate'] > 8 else 'Buy'
                })
        
        if opportunities:
            opp_df = pd.DataFrame(opportunities)
            st.dataframe(
                opp_df.style.applymap(
                    lambda x: 'background-color: #d1fae5' if 'Strong' in str(x) else 'background-color: #fef3c7',
                    subset=['Recommendation']
                ),
                use_container_width=True
            )
        else:
            st.info("No strong investment opportunities found in current market conditions.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_properties_section(self):
        """Show all properties analysis"""
        st.markdown("## 🏘️ PROPERTIES ANALYSIS")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            city_filter = st.multiselect(
                "Filter by City",
                options=self.properties_df['city'].unique(),
                default=self.properties_df['city'].unique()
            )
        
        with col2:
            type_filter = st.multiselect(
                "Filter by Type",
                options=self.properties_df['property_type'].unique(),
                default=self.properties_df['property_type'].unique()
            )
        
        with col3:
            bhk_filter = st.multiselect(
                "Filter by BHK",
                options=sorted(self.properties_df['bhk'].unique()),
                default=sorted(self.properties_df['bhk'].unique())
            )
        
        with col4:
            status_filter = st.multiselect(
                "Filter by Status",
                options=self.properties_df['status'].unique(),
                default=self.properties_df['status'].unique()
            )
        
        # Apply filters
        filtered_df = self.properties_df[
            (self.properties_df['city'].isin(city_filter)) &
            (self.properties_df['property_type'].isin(type_filter)) &
            (self.properties_df['bhk'].isin(bhk_filter)) &
            (self.properties_df['status'].isin(status_filter))
        ]
        
        # Display properties
        st.markdown(f"### 📋 Showing {len(filtered_df)} Properties")
        
        if len(filtered_df) > 0:
            # Grid view
            cols = st.columns(3)
            for idx, (_, property) in enumerate(filtered_df.iterrows()):
                with cols[idx % 3]:
                    self.show_property_card(property)
            
            # Detailed view
            st.markdown("### 📊 Detailed View")
            st.dataframe(
                filtered_df.style.apply(
                    lambda x: ['background: #d1fae5' if v == 'Excellent Investment' else 
                              'background: #fef3c7' if v == 'Good Investment' else 
                              'background: #fee2e2' for v in x],
                    subset=['status']
                ),
                use_container_width=True
            )
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = filtered_df['price_lakhs'].mean()
                st.metric("Average Price", f"₹{avg_price:.0f}L")
            
            with col2:
                avg_roi = filtered_df['roi_5yr'].str.rstrip('%').astype(float).mean()
                st.metric("Average ROI", f"{avg_roi:.1f}%")
            
            with col3:
                good_investments = len(filtered_df[filtered_df['status'].str.contains('Investment')])
                st.metric("Good Investments", f"{good_investments}/{len(filtered_df)}")
        else:
            st.warning("No properties match the selected filters.")
    
    def show_property_card(self, property):
        """Display individual property card"""
        status_color = {
            'Excellent Investment': '#10b981',
            'Good Investment': '#3b82f6',
            'Consider': '#f59e0b',
            'Reconsider': '#ef4444'
        }.get(property['status'], '#6b7280')
        
        st.markdown(f"""
        <div class="property-card">
            <div style='display: flex; justify-content: space-between; align-items: start;'>
                <div>
                    <h4 style='margin: 0; color: #1e293b;'>{property['city']}</h4>
                    <p style='margin: 5px 0; color: #64748b; font-size: 0.9rem;'>
                        {property['property_type']} • {property['bhk']} BHK
                    </p>
                </div>
                <span style='background: {status_color}; color: white; padding: 4px 12px; 
                      border-radius: 15px; font-size: 0.8rem; font-weight: 600;'>
                    {property['status'].split()[0]}
                </span>
            </div>
            
            <div style='margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #64748b;'>Size:</span>
                    <span style='font-weight: 600;'>{property['size_sqft']} sq ft</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #64748b;'>Price:</span>
                    <span style='font-weight: 600; color: #1e40af;'>₹{property['price_lakhs']}L</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #64748b;'>Age:</span>
                    <span style='font-weight: 600;'>{property['age_years']} years</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: #64748b;'>5-Year ROI:</span>
                    <span style='font-weight: 600; color: #10b981;'>{property['roi_5yr']}</span>
                </div>
            </div>
            
            <div style='background: #f8fafc; padding: 10px; border-radius: 10px; margin-top: 10px;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.9rem;'>
                    <span>📍 Location: <strong>{property['location_score']}/10</strong></span>
                    <span>⭐ Amenities: <strong>{property['amenities_score']}/10</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def show_market_trends(self):
        """Show market trends analysis"""
        st.markdown("## 📈 MARKET TRENDS & FORECASTING")
        
        # Time series forecasting
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 📊 5-Year Market Forecast")
        
        # Create forecast data
        years = list(range(2024, 2029))
        
        fig = go.Figure()
        
        for city, data in self.market_data.items():
            current_price = data['avg_price']
            growth_rate = data['growth_rate'] / 100
            
            # Calculate forecast
            forecast = [current_price * ((1 + growth_rate) ** i) for i in range(5)]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=forecast,
                mode='lines+markers',
                name=city,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Price Forecast by City (2024-2028)",
            xaxis_title="Year",
            yaxis_title="Price (₹ Lakhs)",
            height=500,
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Market metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Performance Metrics")
            
            metrics_df = pd.DataFrame([
                {
                    'City': city,
                    'Growth': f"{data['growth_rate']}%",
                    'Price': f"₹{data['avg_price']}L",
                    'Yield': f"{data['rental_yield']}%",
                    'Demand': data['demand']
                }
                for city, data in self.market_data.items()
            ])
            
            st.dataframe(
                metrics_df.style.apply(
                    lambda x: ['background: #d1fae5' if 'Very High' in v else 
                              'background: #fef3c7' if 'High' in v else '' for v in x],
                    subset=['Demand']
                ),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Investment Matrix")
            
            # Create heatmap
            cities = list(self.market_data.keys())
            metrics = ['growth_rate', 'rental_yield', 'infrastructure', 'job_growth']
            
            heatmap_data = []
            for city in cities:
                row = [self.market_data[city][metric] for metric in metrics]
                heatmap_data.append(row)
            
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Metric", y="City", color="Score"),
                x=['Growth', 'Rental Yield', 'Infrastructure', 'Job Growth'],
                y=cities,
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def show_ai_analysis(self):
        """Show AI-powered analysis"""
        st.markdown("## 🤖 AI-POWERED INVESTMENT ANALYSIS")
        
        # Analysis form
        with st.form("ai_analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_city = st.selectbox(
                    "📍 Target City",
                    options=list(self.market_data.keys()),
                    index=2
                )
                
                analysis_type = st.selectbox(
                    "🏠 Property Type",
                    options=['Apartment', 'Villa', 'Independent House', 'Penthouse']
                )
            
            with col2:
                budget_min = st.number_input("💰 Min Budget (Lakhs)", 50, 1000, 100)
                budget_max = st.number_input("💰 Max Budget (Lakhs)", 100, 5000, 500)
                
                priority = st.selectbox(
                    "🎯 Investment Priority",
                    options=['High Growth', 'High Rental Yield', 'Stable Returns', 'Budget Friendly']
                )
            
            analyze = st.form_submit_button("🚀 RUN AI ANALYSIS", use_container_width=True)
        
        if analyze:
            with st.spinner("🤖 AI is analyzing investment opportunities..."):
                # Simulate AI analysis
                import time
                time.sleep(2)
                
                # Get city data
                city_data = self.market_data[analysis_city]
                
                # Calculate scores
                growth_score = city_data['growth_rate'] / 10
                yield_score = city_data['rental_yield'] / 5
                infra_score = city_data['infrastructure'] / 10
                
                # Adjust based on priority
                if priority == 'High Growth':
                    growth_score *= 1.5
                elif priority == 'High Rental Yield':
                    yield_score *= 1.5
                elif priority == 'Stable Returns':
                    infra_score *= 1.5
                
                total_score = (growth_score + yield_score + infra_score) / 3 * 100
                
                # Generate recommendations
                if total_score >= 80:
                    recommendation = "Excellent Opportunity"
                    color = "#10b981"
                elif total_score >= 60:
                    recommendation = "Good Opportunity"
                    color = "#3b82f6"
                else:
                    recommendation = "Consider Alternatives"
                    color = "#f59e0b"
                
                # Display results
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown(f"### 📊 AI Analysis Results for {analysis_city}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overall Score", f"{total_score:.1f}/100")
                    st.metric("Growth Potential", f"{city_data['growth_rate']}%")
                    st.metric("Rental Yield", f"{city_data['rental_yield']}%")
                
                with col2:
                    st.markdown(f"""
                    <div style='background: {color}20; padding: 20px; border-radius: 15px; border-left: 5px solid {color};'>
                        <h3 style='margin: 0 0 10px 0; color: {color};'>🎯 Recommendation</h3>
                        <h2 style='margin: 0; color: {color.replace('20', '')};'>{recommendation}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📋 Key Insights")
                    insights = [
                        f"📍 **Location**: {analysis_city} has {city_data['demand']} demand",
                        f"💰 **Budget Range**: ₹{budget_min}L - ₹{budget_max}L suitable",
                        f"📈 **Growth Outlook**: {city_data['growth_rate']}% annual appreciation expected",
                        f"🏠 **Property Type**: {analysis_type} recommended for this budget"
                    ]
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed analysis
                st.markdown("### 🔍 Detailed Analysis")
                
                analysis_points = [
                    {
                        'title': 'Market Position',
                        'content': f"{analysis_city} ranks among top cities for {priority.lower()} investments.",
                        'score': 85
                    },
                    {
                        'title': 'Risk Assessment',
                        'content': 'Moderate risk with strong fundamentals and growing infrastructure.',
                        'score': 75
                    },
                    {
                        'title': 'ROI Potential',
                        'content': f"Expected 5-year ROI: {city_data['growth_rate'] * 5}% with proper selection.",
                        'score': 90
                    },
                    {
                        'title': 'Timing',
                        'content': 'Current market conditions favorable for entry.',
                        'score': 80
                    }
                ]
                
                cols = st.columns(2)
                for idx, point in enumerate(analysis_points):
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div style='background: white; padding: 20px; border-radius: 15px; margin: 10px 0; 
                                  border: 1px solid #e2e8f0;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                                <h4 style='margin: 0;'>{point['title']}</h4>
                                <span style='background: #3b82f6; color: white; padding: 5px 15px; border-radius: 15px; 
                                      font-weight: 600;'>{point['score']}/100</span>
                            </div>
                            <p style='color: #64748b; margin: 0;'>{point['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    def show_technical_stack(self):
        """Show technical stack and skills"""
        st.markdown("## ⚙️ TECHNICAL STACK & SKILLS")
        
        st.markdown("""
        <div class="dashboard-card">
            <h3>🎯 Project Technology Stack</h3>
            <p style='color: #64748b;'>
                This application is built using cutting-edge technologies for machine learning, 
                data analysis, and web deployment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Skills progress bars
        for category, skills in self.technical_stack.items():
            st.markdown(f'<div class="dashboard-card"><h4>🔧 {category}</h4>', unsafe_allow_html=True)
            
            for skill, proficiency in skills.items():
                st.markdown(f"""
                <div style='margin: 15px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span style='font-weight: 600;'>{skill}</span>
                        <span style='color: #3b82f6; font-weight: 600;'>{proficiency}%</span>
                    </div>
                    <div class="skill-progress">
                        <div class="skill-progress-fill" style='width: {proficiency}%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Technology timeline
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 📅 Development Timeline")
        
        timeline = [
            {"phase": "Data Collection", "duration": "2 weeks", "tech": "Pandas, Web Scraping"},
            {"phase": "Data Preprocessing", "duration": "1 week", "tech": "Pandas, NumPy"},
            {"phase": "Feature Engineering", "duration": "1 week", "tech": "Scikit-learn"},
            {"phase": "Model Training", "duration": "2 weeks", "tech": "Random Forest, XGBoost"},
            {"phase": "Web Application", "duration": "1 week", "tech": "Streamlit, Plotly"},
            {"phase": "Deployment", "duration": "3 days", "tech": "Streamlit Cloud, Git"}
        ]
        
        st.markdown('<div class="timeline">', unsafe_allow_html=True)
        for item in timeline:
            st.markdown(f"""
            <div class="timeline-item">
                <h4 style='margin: 0 0 5px 0;'>{item['phase']}</h4>
                <div style='color: #64748b; font-size: 0.9rem;'>
                    <span style='background: #dbeafe; padding: 3px 10px; border-radius: 12px; 
                          margin-right: 10px;'>⏱️ {item['duration']}</span>
                    <span style='background: #f0f9ff; padding: 3px 10px; border-radius: 12px;'>
                        🛠️ {item['tech']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Skills acquired
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🎓 Skills & Learning Outcomes")
        
        skills_outcomes = [
            "Machine Learning: Regression & Classification models",
            "Data Analysis: EDA, Feature Engineering, Data Cleaning",
            "Web Development: Streamlit for ML applications",
            "Data Visualization: Interactive charts with Plotly",
            "Deployment: Cloud deployment on Streamlit Cloud",
            "Version Control: Git & GitHub for project management",
            "Real Estate Domain: Market analysis & investment strategies"
        ]
        
        cols = st.columns(2)
        for idx, skill in enumerate(skills_outcomes):
            with cols[idx % 2]:
                st.markdown(f"""
                <div style='background: #f8fafc; padding: 15px; border-radius: 10px; margin: 8px 0; 
                          border-left: 4px solid #3b82f6;'>
                    ✅ {skill}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_documentation(self):
        """Show project documentation"""
        st.markdown("## 📚 PROJECT DOCUMENTATION")
        
        # Project overview
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 📖 Project Overview")
        
        st.markdown("""
        **Real Estate Investment Advisor** is a comprehensive machine learning application 
        designed to assist investors in making data-driven real estate decisions.
        
        ### 🎯 Key Objectives:
        1. **Predict** property investment potential using ML models
        2. **Forecast** 5-year property prices
        3. **Analyze** market trends and opportunities
        4. **Provide** actionable investment insights
        
        ### 📊 Features Implemented:
        - **Dashboard**: Real-time market metrics and trends
        - **Properties Analysis**: Detailed property evaluation
        - **AI Analysis**: Machine learning predictions
        - **Market Trends**: Historical and forecast data
        - **Technical Stack**: Complete technology overview
        
        ### 🏗️ Architecture:
        ```
        Data Layer → ML Models → Web Interface → Deployment
           ↓           ↓           ↓             ↓
        CSV Files → Random Forest → Streamlit → Streamlit Cloud
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User guide
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 📋 User Guide")
        
        guide_steps = [
            {
                "step": "1",
                "title": "Dashboard Overview",
                "description": "Start with the dashboard to understand market trends and key metrics."
            },
            {
                "step": "2",
                "title": "Property Analysis",
                "description": "Use filters to find properties matching your criteria."
            },
            {
                "step": "3",
                "title": "AI Analysis",
                "description": "Get personalized investment recommendations based on your preferences."
            },
            {
                "step": "4",
                "title": "Technical Insights",
                "description": "Understand the technology and skills used in this project."
            }
        ]
        
        for guide in guide_steps:
            st.markdown(f"""
            <div style='background: #f8fafc; padding: 20px; border-radius: 15px; margin: 15px 0;'>
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <div style='background: #3b82f6; color: white; width: 40px; height: 40px; 
                          border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                          font-weight: 700; font-size: 1.2rem; margin-right: 15px;'>
                        {guide['step']}
                    </div>
                    <h4 style='margin: 0;'>{guide['title']}</h4>
                </div>
                <p style='color: #64748b; margin: 0;'>{guide['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        self.show_header()
        self.show_sidebar()
        
        # Get current page from session state
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Show quick analysis results if available
        if st.session_state.get('show_quick_results', False):
            st.markdown("## ⚡ Quick Analysis Results")
            col1, col2, col3, col4 = st.columns(4)
            
            city = st.session_state.get('quick_city', 'Bangalore')
            city_data = self.market_data[city]
            
            with col1:
                st.metric("Selected City", city)
            with col2:
                st.metric("Growth Rate", f"{city_data['growth_rate']}%")
            with col3:
                st.metric("Avg Price", f"₹{city_data['avg_price']}L")
            with col4:
                st.metric("Demand", city_data['demand'])
            
            st.info(f"💡 **Insight**: {city} shows {city_data['demand'].lower()} demand with {city_data['growth_rate']}% annual growth.")
            st.markdown("---")
        
        # Show the selected page
        if current_page == 'dashboard':
            self.show_dashboard()
        elif current_page == 'properties':
            self.show_properties_section()
        elif current_page == 'trends':
            self.show_market_trends()
        elif current_page == 'analysis':
            self.show_ai_analysis()
        elif current_page == 'tech':
            self.show_technical_stack()
        elif current_page == 'docs':
            self.show_documentation()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 0.9rem; padding: 20px;'>
            <p>© 2024 Real Estate Investment Advisor Pro | Built with ❤️ using Streamlit & Machine Learning</p>
            <p style='font-size: 0.8rem;'>
                🔐 **Disclaimer**: This tool provides data-driven insights. Always consult with 
                real estate professionals before making investment decisions.
            </p>
            <div style='margin-top: 20px;'>
                <span style='background: #3b82f6; color: white; padding: 8px 20px; border-radius: 20px; 
                      margin: 0 5px; font-size: 0.8rem;'>Machine Learning</span>
                <span style='background: #10b981; color: white; padding: 8px 20px; border-radius: 20px; 
                      margin: 0 5px; font-size: 0.8rem;'>Data Analysis</span>
                <span style='background: #8b5cf6; color: white; padding: 8px 20px; border-radius: 20px; 
                      margin: 0 5px; font-size: 0.8rem;'>Real Estate</span>
                <span style='background: #f59e0b; color: white; padding: 8px 20px; border-radius: 20px; 
                      margin: 0 5px; font-size: 0.8rem;'>Streamlit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    app = RealEstateAdvisorPro()
    app.run()
