import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Smart Tourist & Hospitality System",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = os.getcwd()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🌍 Smart Tourist System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏛 Tourist Places",
        "🍽 Restaurants",
        "🏨 Hotels",
        "💸 Expense Predictor"
    ]
)

# ==============================================================================
# GLOBAL STYLING OVERRIDES (Cyber Dark & Neon Green Theme)
# ==============================================================================
st.markdown("""
<style>
/* App Background */
.stApp {
    background: #0b0e14;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar styling */
div[data-testid="stSidebar"] {
    background-color: #05070a;
    border-right: 1px solid rgba(0, 255, 102, 0.15);
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700;
}

p, label, span {
    color: #94a3b8 !important;
}

/* Cyber Dark Card Style Button */
.stButton>button {
    background: #111622;
    color: #34d399 !important;
    border-radius: 12px;
    height: 52px;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
    border: 1px solid rgba(0, 255, 102, 0.25);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    transition: all 0.2s ease-in-out;
}

.stButton>button:hover {
    background: #1a2234;
    color: #00ff66 !important;
    border-color: #00ff66;
    transform: translateY(-2px);
}

/* Dark Green-Accented Cards */
.cyber-card {
    background: #111622;
    border: 1px solid rgba(0, 255, 102, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.cyber-card:hover {
    border-color: #00ff66;
    box-shadow: 0 8px 24px rgba(0, 255, 102, 0.2);
    transform: translateY(-3px);
}

/* Custom Neon Pill Badges */
.badge-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}
.badge-rating { background: rgba(255, 215, 0, 0.1); color: #ffd700 !important; border: 1px solid rgba(255, 215, 0, 0.3); }
.badge-cost { background: rgba(0, 255, 102, 0.1); color: #00ff66 !important; border: 1px solid rgba(0, 255, 102, 0.3); }
.badge-type { background: rgba(16, 185, 129, 0.15); color: #34d399 !important; border: 1px solid rgba(16, 185, 129, 0.3); }

/* Input field polish */
div[data-baseweb="select"] > div {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: 🏛 TOURIST PLACES
# ==============================================================================
if page == "🏛 Tourist Places":

    @st.cache_resource
    def load_places_models():
        model_path = os.path.join(BASE_DIR, "recommendation_model.pkl")
        encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
        data_path = os.path.join(BASE_DIR, "places_data.pkl")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        places_df = joblib.load(data_path)
        return model, encoder, places_df

    model, encoder, places_df = load_places_models()

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Algorithm:** Cosine Similarity with Nearest Neighbors")

    # --- Hero Section ---
    left, right = st.columns([2, 1])

    with left:
        st.title("🌍 Smart Tourist Recommendation")
        st.caption("Discover top destinations tailored to your travel style and preferences across India.")

    with right:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/854/854878.png",
            width=130
        )

    st.divider()

    # --- Form Filters ---
    st.subheader("🎯 Personalize Your Preferences")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox(
            "State",
            sorted(places_df["State"].dropna().unique())
        )

    city_list = sorted(
        places_df[places_df["State"] == state]["City"].dropna().unique()
    )

    with col2:
        city = st.selectbox(
            "City",
            city_list
        )

    col3, col4 = st.columns(2)

    with col3:
        place_type = st.selectbox(
            "Place Type",
            sorted(places_df["Type"].dropna().unique())
        )

    with col4:
        season = st.selectbox(
            "Best Time to Visit",
            sorted(places_df["Best Time to visit"].dropna().unique())
        )

    rating = st.slider(
        "Minimum Rating Filter",
        1.0, 5.0, 4.0, 0.1
    )

    col_fee, col_dslr = st.columns(2)

    with col_fee:
        fee = st.number_input("Max Entrance Fee (₹)", value=0)

    with col_dslr:
        dslr = st.selectbox(
            "DSLR Allowed",
            sorted(places_df["DSLR Allowed"].dropna().unique())
        )

    st.write("")
    recommend = st.button("🔍 Generate Recommendations")

    # --- Recommendation Logic ---
    if recommend:
        filtered_df = places_df[places_df["State"] == state].copy()
        city_df = filtered_df[filtered_df["City"] == city]

        if len(city_df) >= 3:
            filtered_df = city_df

        if len(filtered_df) == 0:
            st.error("No tourist places match your selected parameters.")
            st.stop()

        features = [
            "State",
            "City",
            "Type",
            "Google review rating",
            "Entrance Fee in INR",
            "DSLR Allowed",
            "Best Time to visit"
        ]

        X_filtered = encoder.transform(filtered_df[features])

        knn = NearestNeighbors(
            n_neighbors=min(5, len(filtered_df)),
            metric="cosine"
        )
        knn.fit(X_filtered)

        input_df = pd.DataFrame({
            "State": [state],
            "City": [city],
            "Type": [place_type],
            "Google review rating": [rating],
            "Entrance Fee in INR": [fee],
            "DSLR Allowed": [dslr],
            "Best Time to visit": [season]
        })

        input_encoded = encoder.transform(input_df)
        distances, indices = knn.kneighbors(input_encoded)

        recommendations = filtered_df.iloc[indices[0]]

        st.subheader(f"✨ Top Recommended Places in {state}")

        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color:#00ff66 !important;">🏛️ {row['Name']}</h3>
                    <div style="margin-bottom: 12px;">
                        <span class="badge-pill badge-type">🏷️ {row['Type']}</span>
                        <span class="badge-pill badge-rating">⭐ {row['Google review rating']} / 5.0</span>
                        <span class="badge-pill badge-cost">💰 ₹{row['Entrance Fee in INR']}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px; color: #cbd5e1;">
                        <div>📍 <b>Location:</b> {row['City']}, {row['State']}</div>
                        <div>📷 <b>DSLR Allowed:</b> {row['DSLR Allowed']}</div>
                        <div>🌤️ <b>Best Season:</b> {row['Best Time to visit']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# PAGE 2: 🍽 RESTAURANTS
# ==============================================================================
elif page == "🍽 Restaurants":

    @st.cache_resource
    def load_restaurant_data():
        csv_path = os.path.join(BASE_DIR, "datasets", "clean_restaurants.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(BASE_DIR, "clean_restaurants.csv")

        sim_path = os.path.join(BASE_DIR, "models", "restaurant_similarity.pkl")
        if not os.path.exists(sim_path):
            sim_path = os.path.join(BASE_DIR, "restaurant_similarity.pkl")

        vec_path = os.path.join(BASE_DIR, "models", "restaurant_vectorizer.pkl")
        if not os.path.exists(vec_path):
            vec_path = os.path.join(BASE_DIR, "restaurant_vectorizer.pkl")

        restaurant_df = pd.read_csv(csv_path)
        similarity = joblib.load(sim_path)
        vectorizer = joblib.load(vec_path)

        return restaurant_df, similarity, vectorizer

    restaurant_df, similarity, vectorizer = load_restaurant_data()

    # --- Header ---
    st.title("🍽️ Smart Restaurant Finder")
    st.caption("Discover top-rated dining spots tailored to your budget and choice of location.")

    st.divider()

    # --- Inputs ---
    col_city, col_budget = st.columns(2)

    with col_city:
        city = st.selectbox(
            "Select Destination City",
            sorted(restaurant_df["City"].dropna().unique())
        )

    with col_budget:
        budget = st.slider(
            "Maximum Budget (₹)",
            100, 5000, 1000, step=100
        )

    top_n = st.slider(
        "Number of Results",
        1, 10, 5
    )

    st.write("")
    recommend = st.button("🍽️ Find Restaurants")

    def recommend_restaurants(city_name, max_budget, n_results=5):
        filtered = restaurant_df[
            (restaurant_df["City"].str.lower() == city_name.lower()) &
            (restaurant_df["Cost"] <= max_budget)
        ]

        if filtered.empty:
            return None

        filtered = filtered.sort_values(
            by=["Rating", "Votes"],
            ascending=False
        )

        return filtered.head(n_results)

    # --- Display Results ---
    if recommend:
        result = recommend_restaurants(city, budget, top_n)

        if result is None:
            st.error("No restaurants found matching your budget and city criteria.")
        else:
            st.subheader(f"✨ Top {len(result)} Curated Dining Options")

            for _, row in result.iterrows():
                st.markdown(
                    f"""
                    <div class="cyber-card">
                        <h3 style="margin-top:0; color:#00ff66 !important;">🍴 {row['Name']}</h3>
                        <div style="margin-bottom: 12px;">
                            <span class="badge-pill badge-rating">⭐ {row['Rating']}</span>
                            <span class="badge-pill badge-cost">💰 Avg Cost ₹{row['Cost']}</span>
                            <span class="badge-pill badge-type">👍 {row['Votes']} Votes</span>
                        </div>
                        <div style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">
                            📍 <b>Location:</b> {row['Location']}, {row['City']}<br>
                            🍝 <b>Cuisine Specialty:</b> {row['Cuisine']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ==============================================================================
# PAGE 3: 🏨 HOTELS
# ==============================================================================
elif page == "🏨 Hotels":

    @st.cache_resource
    def load_hotel_models():
        model_path = os.path.join(BASE_DIR, "models", "hotelrfr.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, "hotelrfr.pkl")

        data_path = os.path.join(BASE_DIR, "datasets", "hotel.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "hotel.csv")

        model = joblib.load(model_path)
        original_df = pd.read_csv(data_path)

        categorical_cols = [
            "City",
            "Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5",
            "Feature_6", "Feature_7", "Feature_8", "Feature_9"
        ]

        encoded_df = original_df.copy()
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoders[col] = le

        return model, original_df, encoded_df, encoders

    model, original_df, df, encoders = load_hotel_models()

    # --- Sidebar Info ---
    st.sidebar.markdown("---")
    st.sidebar.header("MODEL INFORMATION 📋")
    st.sidebar.subheader("Dataset Details")
    st.sidebar.write(f"- **Total Rows:** {df.shape[0]}")
    st.sidebar.write(f"- **Total Columns:** {df.shape[1]}")
    st.sidebar.write("- **Target:** Hotel Rating")

    if st.sidebar.checkbox("Show Dataset Statistics"):
        st.sidebar.metric("Minimum Rating", f"{original_df['Hotel_Rating'].min():.1f}")
        st.sidebar.metric("Maximum Rating", f"{original_df['Hotel_Rating'].max():.1f}")
        st.sidebar.metric("Average Rating", f"{original_df['Hotel_Rating'].mean():.2f}")

    # --- Header ---
    st.title("🏨 Smart Hotel Recommendation System")
    st.caption("Predict rating scores using Random Forest Regressor and discover matching hotels.")

    st.divider()

    # --- Inputs ---
    st.subheader("🎯 Enter Hotel Details & Amenities")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", encoders["City"].classes_)
        feature1 = st.selectbox("Feature 1", encoders["Feature_1"].classes_)
        feature2 = st.selectbox("Feature 2", encoders["Feature_2"].classes_)
        feature3 = st.selectbox("Feature 3", encoders["Feature_3"].classes_)
        feature4 = st.selectbox("Feature 4", encoders["Feature_4"].classes_)
        feature5 = st.selectbox("Feature 5", encoders["Feature_5"].classes_)

    with col2:
        feature6 = st.selectbox("Feature 6", encoders["Feature_6"].classes_)
        feature7 = st.selectbox("Feature 7", encoders["Feature_7"].classes_)
        feature8 = st.selectbox("Feature 8", encoders["Feature_8"].classes_)
        feature9 = st.selectbox("Feature 9", encoders["Feature_9"].classes_)
        hotel_price = st.number_input("Hotel Price (₹)", min_value=0, value=1000)

    st.write("")
    predict_btn = st.button("🔮 Predict Rating & Recommend Hotels")

    # --- Prediction & Recommendations ---
    if predict_btn:
        input_data = pd.DataFrame({
            "City": [encoders["City"].transform([city])[0]],
            "Feature_1": [encoders["Feature_1"].transform([feature1])[0]],
            "Feature_2": [encoders["Feature_2"].transform([feature2])[0]],
            "Feature_3": [encoders["Feature_3"].transform([feature3])[0]],
            "Feature_4": [encoders["Feature_4"].transform([feature4])[0]],
            "Feature_5": [encoders["Feature_5"].transform([feature5])[0]],
            "Feature_6": [encoders["Feature_6"].transform([feature6])[0]],
            "Feature_7": [encoders["Feature_7"].transform([feature7])[0]],
            "Feature_8": [encoders["Feature_8"].transform([feature8])[0]],
            "Feature_9": [encoders["Feature_9"].transform([feature9])[0]],
            "Hotel_Price": [hotel_price]
        })

        predicted_rating = model.predict(input_data)[0]

        st.markdown(f"""
            <div class="cyber-card" style="text-align: center; border-color: #00ff66;">
                <h3 style="margin:0;">⭐ Predicted Hotel Rating</h3>
                <h1 style="color: #00ff66 !important; font-size: 42px; margin: 10px 0;">{predicted_rating:.2f} / 5.0</h1>
            </div>
        """, unsafe_allow_html=True)

        # Recommendation Logic
        recommendations = original_df[
            (original_df["City"] == city) &
            (original_df["Hotel_Rating"] >= predicted_rating - 0.3)
        ].sort_values("Hotel_Rating", ascending=False)

        if recommendations.empty:
            recommendations = original_df[
                original_df["City"] == city
            ].sort_values("Hotel_Rating", ascending=False).head(5)

        st.subheader("🏨 Recommended Hotels")

        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="cyber-card">
                    <h3 style="margin-top:0; color:#00ff66 !important;">🏨 {row['Hotel_Name']}</h3>
                    <div style="margin-bottom: 12px;">
                        <span class="badge-pill badge-rating">⭐ Rating: {row['Hotel_Rating']}</span>
                        <span class="badge-pill badge-cost">💰 ₹{row['Hotel_Price']} / night</span>
                    </div>
                    <div style="font-size: 14px; color: #cbd5e1;">
                        📍 <b>Location:</b> {row['City']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# PAGE 4: 💸 EXPENSE PREDICTOR
# ==============================================================================
elif page == "💸 Expense Predictor":

    @st.cache_resource
    def load_expense_models():
        model_path = os.path.join(BASE_DIR, "models", "expense_prediction_model.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, "expense_prediction_model.pkl")

        data_path = os.path.join(BASE_DIR, "datasets", "Indian_Travel_Expense_Dataset.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "Indian_Travel_Expense_Dataset.csv")

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        categorical = [
            "Destination",
            "State",
            "Hotel_Type",
            "Season",
            "Travel_Mode"
        ]

        encoders = {}
        for col in categorical:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        return model, df, encoders

    model, df, encoders = load_expense_models()

    # --- Sidebar Information ---
    st.sidebar.markdown("---")
    st.sidebar.header("MODEL INFORMATION 📋")
    st.sidebar.subheader("Dataset Details")
    st.sidebar.write(f"- **Total Rows:** {df.shape[0]}")
    st.sidebar.write(f"- **Total Columns:** {df.shape[1]}")
    st.sidebar.write("- **Target:** Total Expense")

    if st.sidebar.checkbox("Show Dataset Statistics"):
        st.sidebar.metric("Minimum Expense", f"₹{df['Total_Expense'].min():,.0f}")
        st.sidebar.metric("Maximum Expense", f"₹{df['Total_Expense'].max():,.0f}")
        st.sidebar.metric("Average Expense", f"₹{df['Total_Expense'].mean():,.0f}")

        st.sidebar.markdown("---")
        st.sidebar.write("### Feature Ranges")
        st.sidebar.write(f"**Days:** {df['Days'].min()} - {df['Days'].max()}")
        st.sidebar.write(f"**Travelers:** {df['Travelers'].min()} - {df['Travelers'].max()}")
        st.sidebar.write(f"**Hotel Cost:** ₹{df['Hotel_Cost'].min()} - ₹{df['Hotel_Cost'].max()}")
        st.sidebar.write(f"**Food Cost:** ₹{df['Food_Cost'].min()} - ₹{df['Food_Cost'].max()}")
        st.sidebar.write(f"**Transport Cost:** ₹{df['Transport_Cost'].min()} - ₹{df['Transport_Cost'].max()}")
        st.sidebar.write(f"**Activity Cost:** ₹{df['Activity_Cost'].min()} - ₹{df['Activity_Cost'].max()}")
        st.sidebar.write(f"**Shopping Cost:** ₹{df['Shopping_Cost'].min()} - ₹{df['Shopping_Cost'].max()}")

    # --- Header ---
    st.title("✈️ Smart Tourism Expense Predictor")
    st.caption("Estimate total trip expenses using Random Forest Regressor based on travel details and itemized costs.")

    st.divider()

    # --- Inputs ---
    st.subheader("🎯 Enter Trip Details")

    col1, col2 = st.columns(2)

    with col1:
        destination = st.selectbox("Destination", encoders["Destination"].classes_)
        state = st.selectbox("State", encoders["State"].classes_)
        hotel_type = st.selectbox("Hotel Type", encoders["Hotel_Type"].classes_)
        season = st.selectbox("Season", encoders["Season"].classes_)
        days = st.number_input("Days", min_value=int(df["Days"].min()), max_value=int(df["Days"].max()), value=int(df["Days"].min()))
        travelers = st.number_input("Travelers", min_value=int(df["Travelers"].min()), max_value=int(df["Travelers"].max()), value=int(df["Travelers"].min()))

    with col2:
        travel_mode = st.selectbox("Travel Mode", encoders["Travel_Mode"].classes_)
        hotel_cost = st.number_input("Hotel Cost (₹)", min_value=0, max_value=130000, value=1000)
        food_cost = st.number_input("Food Cost (₹)", min_value=0, max_value=43499, value=500)
        transport_cost = st.number_input("Transport Cost (₹)", min_value=0, max_value=10000, value=300)
        activity_cost = st.number_input("Activity Cost (₹)", min_value=0, max_value=20000, value=200)
        shopping_cost = st.number_input("Shopping Cost (₹)", min_value=0, max_value=10000, value=500)

    st.write("")
    predict_expense = st.button("💸 Predict Total Expense")

    # --- Prediction ---
    if predict_expense:
        input_data = pd.DataFrame({
            "Destination": [encoders["Destination"].transform([destination])[0]],
            "State": [encoders["State"].transform([state])[0]],
            "Days": [days],
            "Travelers": [travelers],
            "Hotel_Type": [encoders["Hotel_Type"].transform([hotel_type])[0]],
            "Hotel_Cost": [hotel_cost],
            "Food_Cost": [food_cost],
            "Transport_Cost": [transport_cost],
            "Activity_Cost": [activity_cost],
            "Shopping_Cost": [shopping_cost],
            "Season": [encoders["Season"].transform([season])[0]],
            "Travel_Mode": [encoders["Travel_Mode"].transform([travel_mode])[0]]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(f"""
            <div class="cyber-card" style="text-align: center; border-color: #00ff66;">
                <h3 style="margin:0;">💸 Estimated Total Trip Expense</h3>
                <h1 style="color: #00ff66 !important; font-size: 42px; margin: 10px 0;">₹ {prediction:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.balloons()