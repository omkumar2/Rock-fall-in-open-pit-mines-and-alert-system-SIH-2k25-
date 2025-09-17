
import os
import streamlit as st
import requests
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Rockfall Prediction Dashboard", layout="wide")
st.title("🌄 AI-Based Rockfall Prediction System")
location = st.text_input("Enter Mine Location")
if st.button("Predict Risk"):
    if location:
        api_url = os.getenv("API_URL", "http://localhost:8000/predict")
        payload = {"location": location}
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                risk_level = data['risk_level']
                probability = data['probability']
                latitude = data.get('latitude', 25.0)
                longitude = data.get('longitude', 85.0)
                st.write(f"### 📊 Risk Level: {risk_level}")
                st.write(f"### 🎯 Probability: {probability:.2f}")
                if risk_level == "High":
                    st.error("⚠️ High Rockfall Risk Detected!")
                elif risk_level == "Moderate":
                    st.warning("⚠️ Moderate Rockfall Risk.")
                else:
                    st.success("✅ Rockfall Risk is Stable.")
                color = "green"
                if risk_level == "High":
                    color = "red"
                elif risk_level == "Moderate":
                    color = "orange"
                m = folium.Map(location=[latitude, longitude], zoom_start=10)
                folium.Circle(location=[latitude, longitude], radius=5000, color=color, fill=True, fill_opacity=0.7, popup=f"Risk: {risk_level}").add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.error("❌ API Error: Could not get prediction.")
        except Exception as e:
            st.error(f"❌ API Call failed: {e}")
    else:
        st.warning("⚠️ Please enter a location.")
