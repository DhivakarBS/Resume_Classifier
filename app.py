import streamlit as st
import pickle
import fitz  # PyMuPDF
import re
import pandas as pd
import plotly.express as px

# 1. Load the Brain
classifier = pickle.load(open('classifier.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# --- UI Setup ---
st.set_page_config(page_title="AI Career Path Analyzer", page_icon="📊")
st.title("📊 AI Career Path Analyzer")
st.markdown("Upload your resume to see your professional 'DNA' across different fields.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # PDF Extraction
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = " ".join([page.get_text() for page in doc])
    
    cleaned_resume = clean_text(text)
    input_features = tfidf.transform([cleaned_resume])
    
    # 2. Get Probabilities for ALL categories
    # predict_proba gives the confidence score for every field in the dataset
    probs = classifier.predict_proba(input_features)[0]
    all_categories = encoder.classes_
    
    # Create a DataFrame for visualization
    prob_df = pd.DataFrame({
        'Field': all_categories,
        'Probability': probs * 100
    }).sort_values(by='Probability', ascending=False)

    # 3. Display Top Result
    top_field = prob_df.iloc[0]['Field']
    top_score = prob_df.iloc[0]['Probability']
    
    st.success(f"Primary Identity: **{top_field}** ({top_score:.1f}%)")

    # 4. Show the Probability Chart
    # We only show the top 8 fields to keep the chart clean
    fig = px.bar(prob_df.head(8), x='Probability', y='Field', orientation='h',
                 title="Professional Skill Distribution",
                 color='Probability', color_continuous_scale='Blues')
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)

    # 5. Skill Highlights based on your actual projects
    st.write("### 🛠️ Key Technical Assets Detected:")
    cols = st.columns(2)
    with cols[0]:
        st.write("**AI/ML Strength**")
        # Checking for your specific CV keywords
        if 'tensorflow' in cleaned_resume or 'pytorch' in cleaned_resume:
            st.code("Deep Learning Specialist (BioLens/AURA)")
    with cols[1]:
        st.write("**Software Strength**")
        if 'django' in cleaned_resume or 'html' in cleaned_resume:
            st.code("Full-Stack Capable (Event App)")



# import streamlit as st
# import pickle
# import fitz  # PyMuPDF
# import re

# # 1. Load the "Brain"
# classifier = pickle.load(open('classifier.pkl', 'rb'))
# tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# def clean_text(text):
#     text = re.sub(r'http\S+\s*', ' ', text)
#     text = re.sub(r'RT|cc', ' ', text)
#     text = re.sub(r'#\S+', '', text)
#     text = re.sub(r'@\S+', '  ', text)
#     text = re.sub(r'[^\x00-\x7f]', r' ', text) 
#     text = re.sub(r'\s+', ' ', text)
#     return text.lower()

# # --- UI Setup ---
# st.set_page_config(page_title="AI Resume Insights", page_icon="🔍")
# st.title("🔍 AI Resume Keyword Insights")

# uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# if uploaded_file is not None:
#     with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
#         text = ""
#         for page in doc:
#             text += page.get_text()
    
#     cleaned_resume = clean_text(text)
#     input_features = tfidf.transform([cleaned_resume])
#     score = classifier.predict(input_features)[0]

#     # --- NEW: Keyword Highlighting Logic ---
#     # These are key terms the model likely looks for based on your CV
#     important_keywords = [
#         'python', 'machine learning', 'artificial intelligence', 'data science', 
#         'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 
#         'opencv', 'nlp', 'llm', 'django', 'cnn', 'deep learning'
#     ]
    
#     found_keywords = [word for word in important_keywords if word in cleaned_resume]

#     # --- Display Results ---
#     st.subheader(f"Compatibility Score: {score:.1f}%")
#     st.progress(score / 100 if score <= 100 else 1.0)

#     st.write("### 🔑 Key Skills Detected by AI:")
#     if found_keywords:
#         # Create a cool "Tag" view for found keywords
#         cols = st.columns(len(found_keywords) if len(found_keywords) < 5 else 5)
#         for i, word in enumerate(found_keywords):
#             cols[i % 5].success(word.upper())
#     else:
#         st.error("No major technical keywords detected. The AI might be struggling to read the PDF format.")

#     with st.expander("Why is my score what it is?"):
#         st.write("""
#         The AI looks for specific 'tokens' (words) that appeared frequently in the 30,000 successful 
#         hiring records it was trained on. If your score is low, it might be because the PDF text 
#         extraction is adding 'noise' (like page numbers or formatting symbols) that confuses the model.
#         """)










