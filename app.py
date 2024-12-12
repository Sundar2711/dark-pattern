from flask import Flask, render_template, request
import cloudscraper
from bs4 import BeautifulSoup
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Load the trained model
model = joblib.load('dark_pattern_detection_model_naive_bayes.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def classify_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

def scrape_and_classify(url):
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)
        response.raise_for_status()  # Check for HTTP errors

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        paragraph_text = '\n'.join(paragraph.get_text(separator=' ', strip=True) for paragraph in paragraphs)
        cleaned_text = re.sub(r'\[.*?\]|\(.*?\)|http\S+|\S*@\S*', '', paragraph_text)

        results = []
        for line in cleaned_text.splitlines():
            if line and '.' in line:
                category = classify_text(line)
                results.append((line, category))

        return results

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        url = request.form.get('url')
        results = scrape_and_classify(url)

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
