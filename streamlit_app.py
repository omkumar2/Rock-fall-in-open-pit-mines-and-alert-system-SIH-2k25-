
import os
import streamlit as st
import requests
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Rockfall Prediction Dashboard", layout="wide")
st.title("🌄 AI-Based Rockfall Prediction System")
location = st.text_input("Enter Mine Location")
if st.button("Predict Risk"):
    if not location:
        st.warning("⚠️ Please enter a location.")
    else:
        api_url = os.getenv("API_URL", "http://localhost:8000/predict")
        st.info("🔄 Processing prediction request...")
        
        payload = {"location": location}
        try:
            response = requests.post(api_url, json=payload, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            risk_level = data.get('risk_level')
            probability = data.get('probability')
            latitude = data.get('latitude', 25.0)
            longitude = data.get('longitude', 85.0)
            
            if not all([risk_level, probability is not None]):
                st.error("❌ Invalid response format from API")
                st.json(data)  # Show the actual response for debugging
            else:
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### 📊 Risk Level:")
                    if risk_level == "High":
                        st.error("⚠️ High Rockfall Risk Detected!")
                    elif risk_level == "Moderate":
                        st.warning("⚠️ Moderate Rockfall Risk")
                    else:
                        st.success("✅ Rockfall Risk is Stable")
                    
                with col2:
                    st.write("### 🎯 Probability:")
                    st.write(f"{probability:.2%}")
                
                # Map visualization
                color = {"High": "red", "Moderate": "orange"}.get(risk_level, "green")
                m = folium.Map(location=[latitude, longitude], zoom_start=10)
                folium.Circle(
                    location=[latitude, longitude],
                    radius=5000,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"Risk: {risk_level}\nProbability: {probability:.2%}"
                ).add_to(m)
                st_folium(m, width=700, height=500)
                
        except requests.exceptions.Timeout:
            st.error("⏱️ API request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("📡 Could not connect to the API server. Please check if the server is running.")
            st.info("💡 Tip: Make sure the FastAPI server is running on http://localhost:8000")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'json'):
                st.json(e.response.json())  # Show detailed error response if available
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("Please contact support if this error persists.")
