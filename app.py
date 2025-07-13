import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load models and data
vectorizer = joblib.load("models/vectorizer.pkl")
stacked_model = joblib.load("models/stacked_model.pkl")
df_all = pd.read_csv("data/preprocessed/courses_all_labeled.csv")

# Function to recommend similar courses
def recommend_similar_courses(search_text, role, n=5):
    df_role = df_all[df_all["job_role"] == role].drop_duplicates(subset=["course_title"])
    course_vecs = vectorizer.transform(df_role["course_title"])
    user_vec = vectorizer.transform([search_text])
    similarities = cosine_similarity(user_vec, course_vecs).flatten()
    top_idx = similarities.argsort()[::-1][:n]
    return df_role.iloc[top_idx]["course_title"].tolist()

# GitHub repo fetcher
def search_github_repos(query, per_page=5):
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
    headers = {"Accept": "application/vnd.github+json"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 403:
        st.warning("❌ GitHub rate limit exceeded. Try again later.")
        return []
    elif response.status_code != 200:
        st.error(f"Error: {response.status_code}, {response.json()}")
        return []
    return response.json()["items"]

# --- Streamlit UI ---
st.set_page_config(page_title="Course-to-Career Recommender", layout="centered")

st.title("🎯 Course-to-Career Recommender")
st.markdown("Enter the **recent courses** you’ve completed or enrolled in. One by one:")

# Session state for dynamic input
if "course_list" not in st.session_state:
    st.session_state.course_list = []

# --- Callback to safely handle input and clear ---
def add_course():
    course_clean = st.session_state.new_course_input.strip()
    already_exists = any(c.lower() == course_clean.lower() for c in st.session_state.course_list)

    if course_clean and not already_exists:
        st.session_state.course_list.append(course_clean)
    elif already_exists:
        st.warning("⚠️ You've already added this course.")
    elif not course_clean:
        st.warning("⚠️ Course title cannot be empty.")

    # Clear the input field
    st.session_state.new_course_input = ""

# Input field with safe clearing and duplicate handling
st.text_input("📘 Add a course title", key="new_course_input", on_change=add_course)

# Display entered list
if st.session_state.course_list:
    st.markdown("### ✅ Courses Added:")
    for i, c in enumerate(st.session_state.course_list, 1):
        st.write(f"{i}. {c}")

    if st.button("🔍 Done - Predict My Role"):
        with st.spinner("Analyzing your profile..."):
            search_history = " | ".join(st.session_state.course_list)
            X_user = vectorizer.transform([search_history])
            pred_role = stacked_model.predict(X_user)[0]
            proba = stacked_model.predict_proba(X_user).flatten()
            top_idx = proba.argsort()[::-1][:2]
            top_roles = [stacked_model.classes_[i] for i in top_idx]

            # Output prediction
            st.success(f"🎯 Predicted Role: **{top_roles[0]}**")
            st.info(f"💡 You might also be interested in: **{top_roles[1]}**")

            # Show recommended courses
            st.markdown("### 📚 Recommended Courses")
            recs = recommend_similar_courses(search_history, top_roles[0])
            for r in recs:
                st.write("•", r)

            # GitHub repo recommendations
            st.markdown(f"### 💻 Top GitHub Repositories for **{top_roles[0]}**")
            repos = search_github_repos(f"{top_roles[0]} in:name,description")
            for r in repos:
                st.markdown(f"""
🔹 **[{r['full_name']}]({r['html_url']})**  
⭐ {r['stargazers_count']} stars  
📄 {r['description']}
""")
else:
    st.info("Enter at least one course to begin.")

# Option to reset
if st.button("🔁 Start Over"):
    st.session_state.course_list = []
    st.experimental_rerun()
