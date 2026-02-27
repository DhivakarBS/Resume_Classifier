import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor # We use Regressor to predict the Score

# 1. Load Dataset
df = pd.read_csv('Job Datsset.csv')

# Use Job_Requirements as the text and Match_Score as the target
text_col = 'Job_Requirements'
target_col = 'Match_Score'

# 2. Fast Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df[text_col] = df[text_col].apply(clean_text)

# 3. Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df[text_col])
y = df[target_col]

# 4. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This model will now learn what keywords lead to a high "Match_Score"
model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print(f"Model trained to predict Match Scores!")
print(f"R-squared Score: {model.score(X_test, y_test):.2f}")

# 5. Save the brain
pickle.dump(model, open('classifier.pkl', 'wb')) # We call it classifier.pkl to keep app.py simple
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("Files updated! Your app will now predict 'Match Strength' rather than just a Category.")