"""
Fynd AI Assessment - Task 2: User & Admin Dashboard System
UPDATED: Using Google Gemini API (Free Tier)

Web-based feedback system with AI-generated responses and analytics.

Framework: Streamlit
LLM: Google Gemini API
Data Storage: JSON file
Deployment: Streamlit Cloud / Render / HuggingFace Spaces
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from typing import Dict, List
import google.generativeai as genai

# ============================================================================
# GEMINI API CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY environment variable not set")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ============================================================================
# DATA SERVICE
# ============================================================================

DATA_FILE = "reviews_data.json"

def initialize_data_file():
    """Create data file if it doesn't exist"""
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump({"reviews": []}, f)

def add_review(user_review: str, user_rating: int, ai_response: str, ai_summary: str, ai_actions: str) -> Dict:
    """Add a new review and AI responses to storage"""
    initialize_data_file()
    
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    review_entry = {
        "id": len(data["reviews"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "user_rating": user_rating,
        "user_review": user_review,
        "ai_response": ai_response,
        "ai_summary": ai_summary,
        "ai_recommended_actions": ai_actions,
        "status": "new"
    }
    
    data["reviews"].append(review_entry)
    
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    return review_entry

def get_all_reviews() -> List[Dict]:
    """Retrieve all reviews"""
    initialize_data_file()
    
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    return data["reviews"]

def get_reviews_dataframe() -> pd.DataFrame:
    """Get reviews as pandas DataFrame for analysis"""
    reviews = get_all_reviews()
    if not reviews:
        return pd.DataFrame()
    
    df = pd.DataFrame(reviews)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def update_review_status(review_id: int, status: str) -> bool:
    """Update review status"""
    initialize_data_file()
    
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    for review in data["reviews"]:
        if review["id"] == review_id:
            review["status"] = status
            break
    
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    return True

def get_analytics_summary() -> Dict:
    """Generate analytics summary"""
    df = get_reviews_dataframe()
    
    if df.empty:
        return {
            "total_reviews": 0,
            "average_rating": 0,
            "rating_distribution": {},
            "recent_reviews": []
        }
    
    return {
        "total_reviews": len(df),
        "average_rating": float(df["user_rating"].mean()),
        "rating_distribution": df["user_rating"].value_counts().to_dict(),
        "recent_reviews": df.tail(5).to_dict("records")
    }

# ============================================================================
# LLM SERVICE (GEMINI)
# ============================================================================

def generate_ai_response(user_review: str, user_rating: int) -> str:
    """Generate empathetic AI response to user review"""
    
    tone_map = {
        1: "apologetic and solution-focused",
        2: "empathetic and improvement-focused",
        3: "professional and balanced",
        4: "appreciative and reinforcing",
        5: "enthusiastic and grateful"
    }
    
    tone = tone_map.get(user_rating, "professional")
    
    prompt = f"""You are a customer service AI for a restaurant/business.
    
A customer left the following {user_rating}-star review:
"{user_review}"

Generate a brief, professional, and {tone} response (2-3 sentences max).
Be genuine and specific to their feedback.
Do NOT mention this is AI-generated."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Thank you for your feedback. We appreciate your review!"

def generate_summary(user_review: str) -> str:
    """Generate concise summary of review"""
    
    prompt = f"""Summarize this customer review in 1-2 sentences (max 20 words):
    
"{user_review}"

Output ONLY the summary, no preamble."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Customer feedback received"

def generate_recommended_actions(user_review: str, user_rating: int) -> str:
    """Generate recommended actions for admin based on review"""
    
    if user_rating >= 4:
        action_prompt = "Based on this positive review, suggest 1-2 ways to maintain or improve this experience:"
    elif user_rating == 3:
        action_prompt = "Based on this mixed review, suggest 1-2 specific improvements:"
    else:
        action_prompt = "Based on this negative review, suggest 1-2 urgent actions to address concerns:"
    
    prompt = f"""{action_prompt}
    
Review: "{user_review}"
Rating: {user_rating} stars

Output ONLY the actions as a bullet list (max 2 items, concise)."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "• Review and follow up with customer"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fynd AI Feedback System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "page" not in st.session_state:
    st.session_state.page = "User Dashboard"

initialize_data_file()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🎯 Fynd AI Feedback System")
st.sidebar.markdown("**Powered by Google Gemini API**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Dashboard:",
    ["👤 User Dashboard", "👨‍💼 Admin Dashboard"],
    key="page_selector"
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    ### About This System
    - **User Dashboard**: Submit reviews and receive AI responses
    - **Admin Dashboard**: Monitor feedback, analyze trends
    - **AI-Powered**: All responses generated with Google Gemini
    """
)

# ============================================================================
# USER DASHBOARD
# ============================================================================

if "User Dashboard" in page:
    st.title("👤 Customer Feedback Portal")
    st.markdown("Share your experience and get instant AI-generated response")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Submit Your Review")
        
        # Rating selection
        rating = st.select_slider(
            "Rate your experience:",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: ["😞 Poor", "😕 Fair", "😐 Average", "🙂 Good", "😍 Excellent"][x-1]
        )
        
        # Review text input
        review_text = st.text_area(
            "Share your feedback (min 10 characters):",
            placeholder="Tell us about your experience...",
            height=150
        )
        
        # Submit button
        col_submit, col_clear = st.columns(2)
        
        with col_submit:
            submit_button = st.button("📤 Submit Review", use_container_width=True, type="primary")
        
        with col_clear:
            clear_button = st.button("🔄 Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        # Validation and processing
        if submit_button:
            if len(review_text.strip()) < 10:
                st.error("❌ Please enter at least 10 characters")
            else:
                with st.spinner("🤖 Generating AI response using Gemini..."):
                    try:
                        # Generate AI responses
                        ai_response = generate_ai_response(review_text, rating)
                        ai_summary = generate_summary(review_text)
                        ai_actions = generate_recommended_actions(review_text, rating)
                        
                        # Store in database
                        add_review(
                            user_review=review_text,
                            user_rating=rating,
                            ai_response=ai_response,
                            ai_summary=ai_summary,
                            ai_actions=ai_actions
                        )
                        
                        st.success("✅ Review submitted successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("### Stats")
        reviews = get_all_reviews()
        analytics = get_analytics_summary()
        
        st.metric("Total Reviews", analytics["total_reviews"])
        st.metric("Avg Rating", f"{analytics['average_rating']:.1f}/5" if analytics['average_rating'] > 0 else "N/A")
        
        if analytics['rating_distribution']:
            st.markdown("**Rating Breakdown**")
            for rating, count in sorted(analytics['rating_distribution'].items()):
                stars = "⭐" * rating
                st.write(f"{stars}: {count} reviews")
    
    st.markdown("---")
    st.markdown("### Recent Submissions")
    
    df = get_reviews_dataframe()
    if not df.empty:
        df_display = df[["timestamp", "user_rating", "user_review"]].tail(5).copy()
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        df_display.columns = ["Timestamp", "Rating", "Review"]
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No reviews yet. Be the first to submit!")

# ============================================================================
# ADMIN DASHBOARD
# ============================================================================

elif "Admin Dashboard" in page:
    st.title("👨‍💼 Admin Dashboard")
    st.markdown("Monitor reviews, analyze feedback, and manage customer responses")
    
    # Refresh button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Refresh Data", key="refresh_btn"):
            st.rerun()
    
    # Get data
    reviews = get_all_reviews()
    df = get_reviews_dataframe()
    analytics = get_analytics_summary()
    
    if df.empty:
        st.warning("⚠️ No reviews yet. Check back soon!")
    else:
        # Key Metrics Row
        st.markdown("### 📈 Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Reviews", analytics["total_reviews"])
        
        with metric_col2:
            st.metric("Avg Rating", f"{analytics['average_rating']:.2f}/5")
        
        with metric_col3:
            negative_count = len(df[df["user_rating"] <= 2])
            st.metric("Low Ratings (≤2)", negative_count)
        
        with metric_col4:
            positive_count = len(df[df["user_rating"] >= 4])
            st.metric("High Ratings (≥4)", positive_count)
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Rating Distribution")
            fig_dist = px.bar(
                df["user_rating"].value_counts().sort_index().reset_index(),
                x="user_rating",
                y="count",
                labels={"user_rating": "Rating", "count": "Number of Reviews"},
                color="user_rating",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_chart2:
            st.markdown("### Reviews Over Time")
            df_time = df.set_index("timestamp").resample("D").size()
            fig_time = px.line(
                x=df_time.index,
                y=df_time.values,
                labels={"x": "Date", "y": "Number of Reviews"},
                markers=True
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        st.markdown("---")
        
        # Live Review List
        st.markdown("### 📋 All Submissions (Live Feed)")
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_rating = st.multiselect(
                "Filter by Rating:",
                options=[1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5]
            )
        
        with col_filter2:
            sort_by = st.selectbox(
                "Sort by:",
                options=["Newest First", "Oldest First", "Lowest Rating", "Highest Rating"]
            )
        
        # Apply filters
        df_filtered = df[df["user_rating"].isin(filter_rating)]
        
        if sort_by == "Newest First":
            df_filtered = df_filtered.sort_values("timestamp", ascending=False)
        elif sort_by == "Oldest First":
            df_filtered = df_filtered.sort_values("timestamp", ascending=True)
        elif sort_by == "Lowest Rating":
            df_filtered = df_filtered.sort_values("user_rating", ascending=True)
        elif sort_by == "Highest Rating":
            df_filtered = df_filtered.sort_values("user_rating", ascending=False)
        
        # Display reviews
        for idx, review in df_filtered.iterrows():
            with st.container(border=True):
                col_rating, col_time, col_status = st.columns([1, 2, 1])
                
                with col_rating:
                    rating = int(review["user_rating"])
                    stars = "⭐" * rating
                    st.write(f"**{stars}**")
                
                with col_time:
                    timestamp = review["timestamp"].strftime("%Y-%m-%d %H:%M")
                    st.write(f"📅 {timestamp}")
                
                with col_status:
                    current_status = review.get("status", "new")
                    new_status = st.selectbox(
                        "Status:",
                        options=["new", "reviewed", "resolved"],
                        index=["new", "reviewed", "resolved"].index(current_status),
                        key=f"status_{review['id']}"
                    )
                    if new_status != current_status:
                        update_review_status(review["id"], new_status)
                        st.rerun()
                
                st.write(f"**Customer Review:**")
                st.write(review["user_review"])
                
                col_summary, col_response, col_actions = st.columns(3)
                
                with col_summary:
                    st.write("**AI Summary:**")
                    st.info(review["ai_summary"])
                
                with col_response:
                    st.write("**AI Response (to Customer):**")
                    st.success(review["ai_response"])
                
                with col_actions:
                    st.write("**Recommended Actions:**")
                    st.warning(review["ai_recommended_actions"])
                
                st.divider()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <small>Fynd AI Internship Assessment • Task 2 • Powered by Google Gemini & Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
