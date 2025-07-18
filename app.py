import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .example-button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

class ConcreteStrengthPredictor:
    def __init__(self):
        self.feature_info = {
            'Cement': {'min': 100, 'max': 600, 'unit': 'kg/m³', 'default': 300},
            'FA': {'min': 0, 'max': 400, 'unit': 'kg/m³', 'default': 150},
            'CA': {'min': 600, 'max': 1400, 'unit': 'kg/m³', 'default': 1000},
            'Water': {'min': 100, 'max': 300, 'unit': 'kg/m³', 'default': 180},
            'w/c': {'min': 0.2, 'max': 0.8, 'unit': 'ratio', 'default': 0.5},
            'Curing days': {'min': 1, 'max': 365, 'unit': 'days', 'default': 28},
            'Density ': {'min': 2000, 'max': 2800, 'unit': 'kg/m³', 'default': 2400},
            'Crushing value': {'min': 10, 'max': 40, 'unit': '%', 'default': 25},
            'Water absorption ': {'min': 0.1, 'max': 5.0, 'unit': '%', 'default': 1.5},
            'Abrasion value': {'min': 10, 'max': 50, 'unit': '%', 'default': 30},
            'Specific gravity': {'min': 2.0, 'max': 3.5, 'unit': 'g/cm³', 'default': 2.65}
        }
        
        self.example_mixes = {
            "Standard Mix": {
                'Cement': 350, 'FA': 150, 'CA': 1100, 'Water': 175, 'w/c': 0.5,
                'Curing days': 28, 'Density ': 2400, 'Crushing value': 25,
                'Water absorption ': 1.5, 'Abrasion value': 30, 'Specific gravity': 2.65
            },
            "High Strength Mix": {
                'Cement': 450, 'FA': 200, 'CA': 1000, 'Water': 160, 'w/c': 0.35,
                'Curing days': 56, 'Density ': 2450, 'Crushing value': 20,
                'Water absorption ': 1.2, 'Abrasion value': 25, 'Specific gravity': 2.70
            },
            "Low Cement Mix": {
                'Cement': 250, 'FA': 100, 'CA': 1200, 'Water': 200, 'w/c': 0.6,
                'Curing days': 14, 'Density ': 2350, 'Crushing value': 30,
                'Water absorption ': 2.0, 'Abrasion value': 35, 'Specific gravity': 2.60
            }
        }
        
        self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        """Load the LightGBM model"""
        try:
            # Try to load from different possible locations
            possible_paths = [
                'lightgbm_regressor_model.pkl',
                'model/lightgbm_regressor_model.pkl',
                './lightgbm_regressor_model.pkl'
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("Model file not found. Please upload lightgbm_regressor_model.pkl")
            
            model = joblib.load(model_path)
            
            # Initialize scaler with approximate training data ranges
            scaler = MinMaxScaler()
            dummy_data = []
            for feature, info in _self.feature_info.items():
                dummy_data.append([info['min'], info['max']])
            
            dummy_array = np.array(dummy_data).T
            scaler.fit(dummy_array)
            
            return model, scaler, True
            
        except Exception as e:
            return None, None, False
    
    def predict_strength(self, values):
        """Make prediction"""
        if not self.model_loaded:
            return None, "Model not loaded"
        
        try:
            # Create DataFrame with the input values
            input_data = pd.DataFrame([values])
            
            # Ensure columns are in the same order as training data
            feature_columns = list(self.feature_info.keys())
            input_data = input_data[feature_columns]
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            return prediction, None
            
        except Exception as e:
            return None, str(e)
    
    def get_strength_classification(self, strength):
        """Classify concrete strength"""
        if strength < 20:
            return "Low Strength Concrete", "#ff6b6b"
        elif strength < 40:
            return "Normal Strength Concrete", "#4ecdc4"
        elif strength < 60:
            return "High Strength Concrete", "#45b7d1"
        else:
            return "Very High Strength Concrete", "#96ceb4"
    
    def create_gauge_chart(self, value):
        """Create a gauge chart for the prediction"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Compressive Strength (MPa)"},
            delta = {'reference': 40},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 40], 'color': "gray"},
                    {'range': [40, 60], 'color': "lightgreen"},
                    {'range': [60, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def run_app(self):
        """Run the Streamlit app"""
        
        # Initialize session state for button actions
        if 'action' not in st.session_state:
            st.session_state.action = None
        
        # Load model
        self.model, self.scaler, self.model_loaded = self.load_model()
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏗️ Concrete Compressive Strength Predictor</h1>
            <p>Predict concrete strength using machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please ensure 'lightgbm_regressor_model.pkl' is in the app directory.")
            st.stop()
        else:
            st.success("✅ Model loaded successfully!")
        
        # Sidebar for examples
        st.sidebar.header("📋 Example Concrete Mixes")
        
        for name, values in self.example_mixes.items():
            if st.sidebar.button(name, key=f"sidebar_{name}"):
                st.session_state.action = f"load_{name}"
                st.rerun()
        
        # Process button actions
        if st.session_state.action:
            if st.session_state.action.startswith("load_"):
                mix_name = st.session_state.action.replace("load_", "")
                if mix_name in self.example_mixes:
                    for feature, value in self.example_mixes[mix_name].items():
                        st.session_state[f"default_{feature}"] = value
            elif st.session_state.action == "set_defaults":
                for feature, info in self.feature_info.items():
                    st.session_state[f"default_{feature}"] = info['default']
            elif st.session_state.action == "clear_all":
                for feature in self.feature_info.keys():
                    st.session_state[f"default_{feature}"] = float(self.feature_info[feature]['min'])
            
            # Clear the action
            st.session_state.action = None
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("🏗️ Concrete Mix Parameters")
            
            # First 6 parameters
            input_values = {}
            features = list(self.feature_info.keys())
            
            for feature in features[:6]:
                info = self.feature_info[feature]
                default_value = st.session_state.get(f"default_{feature}", info['default'])
                input_values[feature] = st.number_input(
                    f"{feature} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(default_value),
                    step=0.1 if info['max'] < 10 else 1.0,
                    key=f"input_{feature}",
                    help=f"Range: {info['min']} - {info['max']} {info['unit']}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("📊 Material Properties")
            
            # Last 5 parameters
            for feature in features[6:]:
                info = self.feature_info[feature]
                default_value = st.session_state.get(f"default_{feature}", info['default'])
                input_values[feature] = st.number_input(
                    f"{feature} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(default_value),
                    step=0.1 if info['max'] < 10 else 1.0,
                    key=f"input_{feature}",
                    help=f"Range: {info['min']} - {info['max']} {info['unit']}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔮 Predict Strength", type="primary", use_container_width=True):
                prediction, error = self.predict_strength(input_values)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state['prediction'] = prediction
                    st.session_state['input_values'] = input_values
        
        with col2:
            if st.button("⚙️ Set Defaults", use_container_width=True):
                st.session_state.action = "set_defaults"
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.action = "clear_all"
                st.rerun()
        
        # Display results
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            uncertainty = prediction * 0.1  # 10% uncertainty
            
            # Results card
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Prediction Results</h2>
                <h1 style="font-size: 3rem; margin: 0.5rem 0;">{prediction:.2f} MPa</h1>
                <p style="font-size: 1.2rem; opacity: 0.9;">
                    Range: {prediction-uncertainty:.2f} - {prediction+uncertainty:.2f} MPa
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart and classification
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = self.create_gauge_chart(prediction)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                classification, color = self.get_strength_classification(prediction)
                st.markdown(f"""
                <div style="background: {color}; padding: 2rem; border-radius: 10px; 
                           text-align: center; color: white; margin-top: 2rem;">
                    <h3>🏗️ Classification</h3>
                    <h2>{classification}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Input summary
            st.subheader("📋 Input Parameters Summary")
            
            # Create a DataFrame for display
            summary_data = []
            for feature, value in st.session_state['input_values'].items():
                unit = self.feature_info[feature]['unit']
                summary_data.append({
                    'Parameter': feature,
                    'Value': f"{value:.2f}",
                    'Unit': unit
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Additional info
            st.info("💡 Note: This prediction is based on a machine learning model trained on concrete mix design data. Results should be verified through actual testing.")

# Run the app
if __name__ == "__main__":
    app = ConcreteStrengthPredictor()
    app.run_app()
