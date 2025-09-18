
import os
import streamlit as st
import requests
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Rockfall Prediction Dashboard", layout="wide")
st.title("🌄 AI-Based Rockfall Prediction System")
location = st.text_input("Enter Mine Location", key="location_input")

# Initialize session state for prediction results if not already present
if 'risk_level' not in st.session_state:
    st.session_state['risk_level'] = None
if 'probability' not in st.session_state:
    st.session_state['probability'] = None
if 'latitude' not in st.session_state:
    st.session_state['latitude'] = None
if 'longitude' not in st.session_state:
    st.session_state['longitude'] = None
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False

if st.button("Predict Risk"):
    st.session_state['prediction_made'] = False # Reset prediction status
    if not location:
        st.warning("⚠️ Please enter a location.")
    else:
        st.session_state['prediction_made'] = True # Set prediction status
        api_url = os.getenv("API_URL", "http://localhost:8000/predict")
        st.info("🔄 Processing prediction request...")
        
        payload = {"location": location}
        try:
            response = requests.post(api_url, json=payload, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            st.session_state['risk_level'] = data.get('risk_level')
            st.session_state['probability'] = data.get('probability')
            st.session_state['latitude'] = data.get('latitude', 25.0)
            st.session_state['longitude'] = data.get('longitude', 85.0)
            
            if not all([st.session_state['risk_level'], st.session_state['probability'] is not None]):
                st.error("❌ Invalid response format from API")
                st.json(data)  # Show the actual response for debugging
                st.session_state['prediction_made'] = False # Reset prediction status on error
            
        except requests.exceptions.Timeout:
            st.error("⏱️ API request timed out. Please try again.")
            st.session_state['prediction_made'] = False
        except requests.exceptions.ConnectionError:
            st.error("📡 Could not connect to the API server. Please check if the server is running.")
            st.info("💡 Tip: Make sure the FastAPI server is running on http://localhost:8000")
            st.session_state['prediction_made'] = False
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'json'):
                st.json(e.response.json())  # Show detailed error response if available
            st.session_state['prediction_made'] = False
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("Please contact support if this error persists.")
            st.session_state['prediction_made'] = False

# Display results if a prediction has been made and stored in session state
if st.session_state['prediction_made'] and st.session_state['risk_level'] is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📊 Risk Level:")
        if st.session_state['risk_level'] == "High":
            st.error("⚠️ High Rockfall Risk Detected!")
        elif st.session_state['risk_level'] == "Moderate":
            st.warning("⚠️ Moderate Rockfall Risk")
        else:
            st.success("✅ Rockfall Risk is Stable")
        
    with col2:
        st.write("### 🎯 Probability:")
        st.write(f"{st.session_state['probability']:.2%}")
    
    # Map visualization
    color = {"High": "red", "Moderate": "orange"}.get(st.session_state['risk_level'], "green")
    m = folium.Map(location=[st.session_state['latitude'], st.session_state['longitude']], zoom_start=10)
    folium.Circle(
        location=[st.session_state['latitude'], st.session_state['longitude']],
        radius=5000,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Risk: {st.session_state['risk_level']}\nProbability: {st.session_state['probability']:.2%}"
    ).add_to(m)
    st_folium(m, width=700, height=500)
