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
    initial_sidebar_state="collapsed" # Set to collapsed or removed entirely if not needed
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
    /* Removed .example-button as it's no longer used */
</style>
""", unsafe_allow_html=True)

# Move the cached function outside the class
@st.cache_resource
def load_model():
    """Load the LightGBM model"""
    try:
        # Feature info for scaler initialization
        feature_info = {
            'Cement': {'min': 100 * 0.5, 'max': 600 * 2},
            'FA': {'min': 0 * 0.5, 'max': 400 * 2},
            'CA': {'min': 600 * 0.5, 'max': 1400 * 2},
            'Water': {'min': 100 * 0.5, 'max': 300 * 2},
            'w/c': {'min': 0.2 * 0.5, 'max': 0.8 * 2},
            'Curing days': {'min': 1 * 0.5, 'max': 365 * 2},
            'Density ': {'min': 2000 * 0.5, 'max': 2800 * 2},
            'Crushing value': {'min': 10 * 0.5, 'max': 40 * 2},
            'Water absorption ': {'min': 0.1 * 0.5, 'max': 5.0 * 2},
            'Abrasion value': {'min': 10 * 0.5, 'max': 50 * 2},
            'Specific gravity': {'min': 2.0 * 0.5, 'max': 3.5 * 2}
        }
        
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
        for feature, info in feature_info.items():
            dummy_data.append([info['min'], info['max']])
        
        dummy_array = np.array(dummy_data).T
        scaler.fit(dummy_array)
        
        return model, scaler, True
        
    except Exception as e:
        return None, None, False

class ConcreteStrengthPredictor:
    def __init__(self):
        self.feature_info = {
            'Cement': {'min': 100 * 0.5, 'max': 600 * 2, 'unit': 'kg/m³', 'default': 300},
            'FA': {'min': 0 * 0.5, 'max': 400 * 2, 'unit': 'kg/m³', 'default': 150},
            'CA': {'min': 600 * 0.5, 'max': 1400 * 2, 'unit': 'kg/m³', 'default': 1000},
            'Water': {'min': 100 * 0.5, 'max': 300 * 2, 'unit': 'kg/m³', 'default': 180},
            'w/c': {'min': 0.2 * 0.5, 'max': 0.8 * 2, 'unit': 'ratio', 'default': 0.5},
            'Curing days': {'min': 1 * 0.5, 'max': 365 * 2, 'unit': 'days', 'default': 28},
            'Density ': {'min': 2000 * 0.5, 'max': 2800 * 2, 'unit': 'kg/m³', 'default': 2400},
            'Crushing value': {'min': 10 * 0.5, 'max': 40 * 2, 'unit': '%', 'default': 25},
            'Water absorption ': {'min': 0.1 * 0.5, 'max': 5.0 * 2, 'unit': '%', 'default': 1.5},
            'Abrasion value': {'min': 10 * 0.5, 'max': 50 * 2, 'unit': '%', 'default': 30},
            'Specific gravity': {'min': 2.0 * 0.5, 'max': 3.5 * 2, 'unit': 'g/cm³', 'default': 2.65}
        }
        
        # example_mixes is no longer needed as sidebar buttons are removed
        self.example_mixes = {} 
        
        # Load model using the cached function
        self.model, self.scaler, self.model_loaded = load_model()
    
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
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'input_values' not in st.session_state:
            st.session_state.input_values = {}
            for feature, info in self.feature_info.items():
                st.session_state.input_values[feature] = info['default']
        
        # 'force_update' is no longer needed without sidebar buttons
        # if 'force_update' not in st.session_state:
        #     st.session_state.force_update = False
    
    # update_input_values is no longer needed without sidebar buttons
    # def update_input_values(self, new_values):
    #     """Update input values and force widget refresh"""
    #     st.session_state.input_values = new_values.copy()
    #     st.session_state.force_update = True
    
    def run_app(self):
        """Run the Streamlit app"""
        
        # Initialize session state
        self.initialize_session_state()
        
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
                # Use session state value if available, otherwise use default
                current_value = st.session_state.input_values.get(feature, info['default'])
                
                input_values[feature] = st.number_input(
                    f"{feature} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(current_value),
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
                # Use session state value if available, otherwise use default
                current_value = st.session_state.input_values.get(feature, info['default'])
                
                input_values[feature] = st.number_input(
                    f"{feature} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(current_value),
                    step=0.1 if info['max'] < 10 else 1.0,
                    key=f"input_{feature}",
                    help=f"Range: {info['min']} - {info['max']} {info['unit']}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state with current input values
        st.session_state.input_values = input_values
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔮 Predict Strength", type="primary", use_container_width=True):
                prediction, error = self.predict_strength(input_values)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state['prediction'] = prediction
                    st.session_state['prediction_input_values'] = input_values
        
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
            for feature, value in st.session_state.get('prediction_input_values', {}).items():
                unit = self.feature_info[feature]['unit']
                summary_data.append({
                    'Parameter': feature,
                    'Value': f"{value:.2f}",
                    'Unit': unit
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            # Additional info
            st.info("💡 Note: This prediction is based on a machine learning model trained on concrete mix design data. Results should be verified through actual testing.")

# Run the app
if __name__ == "__main__":
    app = ConcreteStrengthPredictor()
    app.run_app()
