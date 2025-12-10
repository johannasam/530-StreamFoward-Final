import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import numpy as np
import os
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import datetime

load_dotenv()

st.set_page_config(
    page_title="StreamForward | Reflective Log",
    layout="wide"
)

# Green Theme Page Styling 
st.markdown("""
<style>
    /* Import Poppins Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Background - Soft Sage/Mint Cream */
    .stApp {
        background-color: #F0F7F4;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-weight: 600; 
        color: #2F4F4F; /* Dark Slate Gray */
        font-size: 4rem; /* Larger Title */
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #556B2F; /* Dark Olive Green */
        font-size: 1.4rem;
        font-style: italic;
    }

    /* Card Container Style */
    .metric-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
        text-align: center;
        border: 2px solid #E8F5E9; /* Light Green Border */
    }
    
    .metric-card h3 {
        margin: 0;
        color: #8FBC8F; /* Dark Sea Green */
        font-size: 1.1rem; /* Larger label */
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-card h2 {
        margin: 10px 0;
        color: #2F4F4F;
        font-size: 2.5rem; /* Larger Number */
        font-weight: 700;
    }
    
    /* Delta Badges */
    .delta {
        font-size: 1rem;
        font-weight: 500;
        padding: 8px 15px;
        border-radius: 15px;
        display: inline-block;
    }
    .positive { background-color: #C8E6C9; color: #2E7D32; } /* Green 100/800 */
    .negative { background-color: #FFCCBC; color: #BF360C; } /* Deep Orange 100/900 */
    .neutral  { background-color: #F5F5F5; color: #757575; } 
    
    /* Insight Box */
    .insight-box {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 8px solid #8FBC8F; 
        height: 100%;
        font-size: 1.2rem; /* Larger Text */
    }
    
    /* Interpretation Box (New) */
    .interpret-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 15px;
        margin-top: 15px;
        color: #1B5E20;
        font-size: 1.1rem;
    }

    /* Section Headers */
    .section-header {
        color: #2F4F4F;
        font-weight: 600;
        margin-top: 50px;
        margin-bottom: 25px;
        font-size: 1.8rem;
        border-bottom: 3px solid #A5D6A7;
        padding-bottom: 10px;
    }
    
    /* Streak Banner */
    .streak-container {
        background-color: #2F4F4F;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

CSV_DATA = """Country,Age,Gender,Exercise Level,Diet Type,Sleep Hours,Stress Level,Mental Health Condition,Work Hours per Week,Screen Time per Day (Hours),Social Interaction Score,Happiness Score
Brazil,48,Male,Low,Vegetarian,6.3,Low,Mentally Fit,21,4.0,7.8,6.5
Australia,31,Male,Moderate,Vegan,4.9,Low,PTSD,48,5.2,8.2,6.8
Japan,37,Female,Low,Vegetarian,7.2,High,Mentally Fit,43,4.7,9.6,9.7
Brazil,35,Male,Low,Vegan,7.2,Low,Depression,43,2.2,8.2,6.6
Germany,46,Male,Low,Balanced,7.3,Low,Anxiety,35,3.6,4.7,4.4
Japan,23,Other,Moderate,Balanced,2.7,Moderate,Anxiety,50,3.3,8.4,7.2
Japan,49,Male,Moderate,Junk Food,6.6,Low,Anxiety,28,7.2,5.6,6.9
Brazil,46,Other,Low,Vegetarian,6.3,High,PTSD,46,5.6,3.5,1.1
India,60,Male,High,Vegetarian,4.7,Low,Anxiety,33,6.6,3.7,5.2
Germany,19,Female,Moderate,Vegan,3.3,Low,PTSD,25,3.5,4.5,5.5
Canada,41,Other,High,Vegetarian,7.6,High,PTSD,58,4.9,5.7,8.5
Japan,22,Male,Low,Junk Food,8.6,High,Bipolar,42,5.3,5.3,6.4
Australia,23,Female,High,Junk Food,7.2,Low,Bipolar,55,6.2,2.9,9.4
Germany,60,Male,High,Balanced,6.4,Low,Bipolar,51,7.8,7.8,8.8
Germany,61,Other,High,Junk Food,6.3,High,Mentally Fit,23,6.7,2.3,1.7
USA,64,Female,Moderate,Junk Food,6.6,Moderate,Depression,49,3.3,5.4,2.2
Canada,40,Other,High,Balanced,4.8,Moderate,PTSD,58,3.8,4.8,6.4
USA,21,Other,Low,Balanced,7.4,Moderate,Anxiety,24,2.6,6.9,5.7
Japan,44,Male,Moderate,Vegetarian,4.9,Moderate,PTSD,22,5.9,9.3,2.5
Japan,35,Male,Moderate,Keto,8.2,Moderate,PTSD,41,3.1,5.5,6.2"""

# AI Reflective Insights
@st.cache_data
def load_initial_data():
    df = pd.read_csv(io.StringIO(CSV_DATA))
    df['Day'] = range(1, len(df) + 1)
    return df

if 'df' not in st.session_state:
    st.session_state.df = load_initial_data()

def call_llm(prompt, model='GPT 4.1 Mini', max_tokens=200):
    api_key = os.getenv("LITELLM_TOKEN")
    if not api_key:
        raise ValueError("LITELLM_TOKEN not found. Please set it first.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://litellm.oit.duke.edu/v1"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def train_influence_model(df):
    ml_df = df.copy()
    le = LabelEncoder()

    categorical_cols = ['Exercise Level', 'Diet Type', 'Stress Level']
    for col in categorical_cols:
        if col in ml_df.columns:
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            
    features = ['Sleep Hours', 'Work Hours per Week', 'Screen Time per Day (Hours)', 
                'Social Interaction Score', 'Exercise Level']
    target = 'Happiness Score'
    
    ml_df = ml_df.dropna(subset=features + [target])
    X = ml_df[features]
    y = ml_df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'Habit': features,
        'Influence Score': model.feature_importances_
    }).sort_values(by='Influence Score', ascending=False)
    
    return importance

def generate_empathetic_insight(df, current_entry):
    try:
        prompt = (
            f"You are an empathetic wellness companion. The user logged their daily stats: "
            f"Sleep: {current_entry['Sleep Hours']}h (Avg: {df['Sleep Hours'].mean():.1f}h), "
            f"Social Interaction: {current_entry['Social Interaction Score']}/10 (Avg: {df['Social Interaction Score'].mean():.1f}), "
            f"Stress Level: {current_entry['Stress Level']}. "
            f"Give a brief, supportive, and reflective insight (max 2 sentences) about their balance. "
            f"Be supportive but NOT therapeutic: do not diagnose or give medical advice."
        )
        return call_llm(prompt)
            
    except Exception as e:
        return f"API Error: {str(e)}"

# Streak Track/ Daily Log

# Streak Counter
current_streak = len(st.session_state.df) 
st.sidebar.markdown(f"""
<div class="streak-container">
     {current_streak} Day Streak
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Daily Log")
st.sidebar.markdown("_Track your flow to find your balance._")

# Daily Log
with st.sidebar.form(key='daily_log_form'):
    st.markdown("### Today's Rhythm")
    input_sleep = st.slider("Sleep (Hours)", 0.0, 12.0, 7.0, 0.1)
    input_screen = st.slider("Screen Time (Hours)", 0.0, 16.0, 4.0, 0.1)
    input_work = st.number_input("Work Hours", 0, 16, 8)
    input_exercise = st.select_slider("Exercise Level", options=["Low", "Moderate", "High"], value="Moderate")
    input_social = st.slider("Social Interaction", 1, 10, 5)
    
    st.markdown("### Reflection")
    input_stress = st.select_slider("Stress Level", options=["Low", "Moderate", "High"], value="Moderate")
    input_happiness = st.slider("Happiness Score", 1, 10, 5)
    
    input_country, input_gender, input_diet = "User", "Other", "Balanced"
    input_condition, input_age = "None", 30

    submit_button = st.form_submit_button(label='Log Entry')

    if submit_button:
        next_day = st.session_state.df['Day'].max() + 1
        new_data = {
            'Country': input_country, 'Age': input_age, 'Gender': input_gender,
            'Exercise Level': input_exercise, 'Diet Type': input_diet,
            'Sleep Hours': input_sleep, 'Stress Level': input_stress,
            'Mental Health Condition': input_condition,
            'Work Hours per Week': input_work * 5,
            'Screen Time per Day (Hours)': input_screen,
            'Social Interaction Score': input_social,
            'Happiness Score': input_happiness,
            'Day': next_day
        }
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_data])], ignore_index=True)
        st.rerun()

st.sidebar.markdown("---")

# Log Summary and Data Visualization

df = st.session_state.df
last_entry = df.iloc[-1]
avg_data = df.mean(numeric_only=True)

st.markdown("""
<div class="main-header">
    <h1>StreamForward</h1>
    <p>Moving towards a balanced mind, one day at a time.</p>
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)

metrics = [
    ("Sleep Hours", last_entry['Sleep Hours'], avg_data['Sleep Hours'], "Hours"),
    ("Social Score", last_entry['Social Interaction Score'], avg_data['Social Interaction Score'], "/ 10"),
    ("Screen Time", last_entry['Screen Time per Day (Hours)'], avg_data['Screen Time per Day (Hours)'], "Hours"),
    ("Happiness", last_entry['Happiness Score'], avg_data['Happiness Score'], "/ 10")
]

for col, (label, value, avg, unit) in zip(cols, metrics):
    delta = value - avg
    if "Screen" in label:
         delta_class = "positive" if delta < 0 else "negative"
    else:
         delta_class = "positive" if delta > 0 else "negative"
         
    if abs(delta) < 0.2: delta_class = "neutral"
    
    arrow = "↑" if delta > 0 else "↓"
    
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h2>{value} <span style="font-size:1.2rem; color:#8FBC8F;">{unit}</span></h2>
            <div class="delta {delta_class}">
                {arrow} {abs(delta):.1f} vs avg
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Graphs and Explanations

col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown('<div class="section-header">Flow Over Time</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Day'], y=df['Happiness Score'], name='Happiness', 
                             line=dict(color='#8FBC8F', width=5), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df['Day'], y=df['Sleep Hours'], name='Sleep', 
                             line=dict(color='#556B2F', width=2, dash='dot')))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.1, font=dict(size=14)),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        yaxis=dict(gridcolor='#E8F5E9', tickfont=dict(size=14)),
        xaxis=dict(gridcolor='#E8F5E9', tickfont=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown('<div class="section-header">Gentle Reflection</div>', unsafe_allow_html=True)
    insight = generate_empathetic_insight(df, last_entry)
    
    st.markdown(f"""
    <div class="insight-box">
        <strong style="color:#2F4F4F; font-size:1.4rem;">Thinking of you...</strong><br><br>
        <span style="font-style:italic; color:#556B2F; line-height:1.6;">"{insight}"</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-header">Understanding Your Balance</div>', unsafe_allow_html=True)

col_b1, col_b2 = st.columns(2)

with col_b1:
    importance_df = train_influence_model(df)
    top_factor = importance_df.iloc[0]['Habit']
    
    fig_imp = px.bar(
        importance_df, 
        x='Influence Score', 
        y='Habit', 
        orientation='h',
        color='Influence Score',
        color_continuous_scale=['#DCEDC8', '#8FBC8F', '#556B2F'] 
    )
    fig_imp.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=20, b=20), 
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(size=14)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Interpretation Graph 1
    st.markdown(f"""
    <div class="interpret-box">
        <strong>What this means:</strong> This chart calculates which of your daily habits has the strongest statistical connection to your Happiness Score.<br><br>
        <strong>Insight:</strong> Currently, <u>{top_factor}</u> is the biggest driver of your mood. Try keeping this habit consistent this week.
    </div>
    """, unsafe_allow_html=True)

with col_b2:
    fig_scatter = px.scatter(
        df, 
        x='Social Interaction Score', 
        y='Happiness Score', 
        size='Age', 
        color='Stress Level',
        color_discrete_map={'Low': '#8FBC8F', 'Moderate': '#FFEB3B', 'High': '#FF7043'},
        title="Social Connection vs. Happiness"
    )
    fig_scatter.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=40, b=20), 
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(size=14)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Interpretation Graph 2
    st.markdown(f"""
    <div class="interpret-box">
        <strong>What this means:</strong> This maps your social interactions against your happiness.<br><br>
        <strong>Recommendation:</strong> Aim for the top-right corner (High Social + High Happiness). If you see dots in the bottom right, social interaction might be draining you today.
    </div>
    """, unsafe_allow_html=True)