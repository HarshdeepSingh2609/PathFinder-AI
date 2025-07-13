
# 🚀 PathFinder-AI

**PathFinder-AI** is an intelligent course-to-career recommender system that helps users identify their most suitable tech role (e.g., ML Engineer, Data Analyst, Full Stack Developer) based on the courses they've taken — and suggests curated courses and GitHub repositories to accelerate their career path.

---

## 🧠 How It Works

PathFinder-AI uses a **stacked ensemble machine learning model** trained on synthetic user profiles. Here's how it works:

1. **Input**: User enters courses they've completed or are currently enrolled in.
2. **Prediction**: ML model predicts the most likely job role.
3. **Recommendation**:
   - Curated courses to improve for that role
   - Top GitHub repositories related to the predicted role

---

## 🧰 Features

✅ Predicts role from course history  
✅ Suggests additional courses for growth  
✅ Recommends trending GitHub projects  
✅ Clean, interactive interface using **Streamlit**  
✅ Easily extendable with new roles/courses

---

---

## 🧪 Model Details

- **Approach**: Text-based role prediction from course titles
- **Techniques Used**: TF-IDF, Logistic Regression, Random Forest, Stacked Ensembles
- **Synthetic Data**: User-course-role combinations generated with noise to simulate real behavior
- **Evaluation Accuracy**: `97%` on 12 target roles

📌 **Supported Roles**:
- Data Analyst  
- Data Scientist  
- ML Engineer  
- Data Engineer  
- Backend Developer  
- Frontend Developer  
- Full Stack Developer  
- Cloud Engineer  
- DevOps Engineer  
- Cybersecurity Analyst  
- Mobile App Developer  
- UI/UX Designer

