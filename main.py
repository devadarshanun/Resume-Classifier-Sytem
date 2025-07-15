import os
import fitz  # PyMuPDF
import re
import nltk
import spacy
import shutil
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Ensure script uses the correct directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Preprocess the text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_stop and len(token.lemma_) > 2]
    return " ".join(tokens)

# Load and preprocess all resumes
def load_and_clean_resumes(base_path="resumes"):
    texts, labels = [], []
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            if filename.endswith(".pdf"):
                try:
                    file_path = os.path.join(category_path, filename)
                    raw_text = extract_text_from_pdf(file_path)
                    cleaned = preprocess(raw_text)
                    texts.append(cleaned)
                    labels.append(category)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    return texts, labels

# Train and save the model
def train_model():
    texts, labels = load_and_clean_resumes(os.path.join(BASE_DIR, "resumes"))
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, preds))
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/resume_classifier_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print(" Model and vectorizer saved in 'models/' folder")

# Classify a new resume
def classify_resume(file_path):
    model = joblib.load(os.path.join(BASE_DIR, "models/resume_classifier_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl"))
    raw_text = extract_text_from_pdf(file_path)
    cleaned = preprocess(raw_text)
    features = vectorizer.transform([cleaned])
    category = model.predict(features)[0]
    dest_dir = os.path.join(BASE_DIR, f"classified_resumes/{category}")
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(BASE_DIR, "models/resume_classifier_model.pkl"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl"))
    shutil.copy(file_path, dest_dir)
    return category

# GUI interface
def launch_gui():
    def upload_and_classify():
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            try:
                result = classify_resume(file_path)
                messagebox.showinfo("Success", f"Resume classified as: {result}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Resume Classifier")
    root.geometry("400x200")
    tk.Label(root, text="Resume Classifier", font=("Arial", 16)).pack(pady=20)
    tk.Button(root, text="Upload Resume", command=upload_and_classify, font=("Arial", 12), bg="lightblue").pack(pady=10)
    tk.Label(root, text="Files will be saved in classified_resumes/ folder", font=("Arial", 10)).pack(pady=20)
    root.mainloop()

# Main flow
if __name__ == "__main__":
    print(" Step 1: Training the model...")
    train_model()
    print(" Step 2: Launching the GUI...")
    launch_gui()
