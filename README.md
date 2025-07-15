# Resume-Classifier-Sytem
A Python-based application that automatically classifies resumes into categories such as **Engineering**, **Finance**, **Marketing**, etc., using Natural Language Processing (NLP) and machine learning techniques.


## Dataset Information (Not Included in Repository)

Due to the large size of the dataset and numerous resume files (PDFs), the dataset is **not uploaded to this GitHub repository**.

### Dataset Contents
The dataset typically includes:
- A collection of resumes in **PDF format**, stored in a folder like `resumes/`
- A labeled CSV file, e.g. `resume_data.csv`, that contains extracted text and its associated category.

Example row from `resume_data.csv`:
csv
text,category
"Experienced Java developer with Spring Boot experience...",Engineering
"Managed marketing campaigns for international brands...",Marketing

# Requirements
scikit-learn==1.4.1
pandas==2.2.1
numpy==1.26.4
nltk==3.8.1
spacy==3.7.2
PyMuPDF==1.23.19
joblib==1.4.0
tk  # built-in with Python on most platforms, included here for clarity

# Resume Classifier Requirements
Install with: pip install -r requirements.txt

Make sure to run the following after installing:
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
