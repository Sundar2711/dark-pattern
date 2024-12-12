import tkinter as tk
from tkinter import scrolledtext
import cloudscraper
from bs4 import BeautifulSoup
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

ki = []
ji = []

# Load the trained model
model = joblib.load('dark_pattern_detection_model_naive_bayes.pkl')
vectorizer_filename = 'tfidf_vectorizer.pkl'
tfidf_vectorizer = joblib.load(vectorizer_filename)

def classify_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

def scrape_and_classify(url):
    try:
        global ki, ji
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)
        response.raise_for_status()  # Check for HTTP errors

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content from <p> tags
        paragraphs = soup.find_all('p')
        paragraph_text = '\n'.join(paragraph.get_text(separator=' ', strip=True) for paragraph in paragraphs)
        cleaned_text = re.sub(r'\[.*?\]|\(.*?\)|http\S+|\S*@\S*', '', paragraph_text)
        
        with open("sc.txt", 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

        g = ""
        for line in cleaned_text.splitlines():  # Split the cleaned text into lines
            if '.' not in line:
                g += line
            else:
                ji.append(classify_text(g))
                ki.append(g)
                g = ""

        return ki, ji

    except Exception as e:  # Catch the specific exception
        print(f"Error occurred: {e}")  # Print the error message for debugging
        return [], []  # Return empty lists on error

def analyze_url():
    url = entry.get()
    lines, results = scrape_and_classify(url)

    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    if lines:
        for line, result in zip(lines, results):
            result_text.insert(tk.END, f"{line}\nPredicted Category: {result}\n\n")
    else:
        result_text.insert(tk.END, "Error: Unable to scrape content from the provided URL.")
    result_text.config(state=tk.DISABLED)

# Create the main Tkinter window
root = tk.Tk()
root.title("Dark Pattern Detection")

# Create GUI components
url_label = tk.Label(root, text="Enter URL:")
entry = tk.Entry(root, width=40)
analyze_button = tk.Button(root, text="Analyze", command=analyze_url)
result_text = scrolledtext.ScrolledText(root, width=80, height=20, wrap=tk.WORD, state=tk.DISABLED)

# Place GUI components on the window
url_label.grid(row=0, column=0, padx=10, pady=10)
entry.grid(row=0, column=1, padx=10, pady=10)
analyze_button.grid(row=0, column=2, padx=10, pady=10)
result_text.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()
