import streamlit as st
import pandas as pd
import pickle
import time
from src.config import DB_PATH, LOG_PATH

st.set_page_config(page_title="EventGuard Ops", layout="wide")

def get_metrics():
    # 1. Get Total Invited
    total_invited = 0
    if DB_PATH.exists():
        try:
            with open(DB_PATH, "rb") as f:
                total_invited = len(pickle.load(f)["names"])
        except Exception:
            pass # Keep 0 if file is corrupt/empty
            
    # 2. Get Live Logs (The Fix for EmptyDataError)
    df = pd.DataFrame(columns=["Name", "Time", "Status"]) # Default empty
    
    if LOG_PATH.exists():
        try:
            # Check if file has size > 0
            if LOG_PATH.stat().st_size > 0:
                df = pd.read_csv(LOG_PATH)
            else:
                # File exists but is empty -> Return empty DF
                pass 
        except pd.errors.EmptyDataError:
            # Catch the specific error just in case
            pass
            
    return total_invited, len(df), df

st.title("🛡️ EventGuard Command Center")
placeholder = st.empty()

while True:
    total, inside, df = get_metrics()
    
    with placeholder.container():
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Guest List", total)
        
        # Avoid division by zero
        percentage = f"{inside/total:.1%}" if total > 0 else "0%"
        m2.metric("Checked In", inside, delta=percentage)
        
        m3.metric("Remaining", total - inside)

        st.divider()
        st.subheader("Live Access Feed")
        
        # Display DataFrame (Reverse order to see newest first)
        st.dataframe(
            df.sort_index(ascending=False), 
            use_container_width=True,
            hide_index=True
        )
    
    time.sleep(1)
