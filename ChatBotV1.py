import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words
from collections import Counter
import tkinter as tk

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

def record_basic_words(text):
    word_list = words.words()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    basic_words = [word.lower() for word in tokens if word.lower() in word_list]
    word_counts = Counter(basic_words)
    return word_counts

def classify_words_by_occurrence(word_counts):
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    words = [word for word, _ in sorted_words]
    return " ".join(words)

def process_input():
    user_input = entry.get()
    if user_input.lower() == "exit":
        exit_program()
    else:
        response = generate_response(user_input)
        display_response(response)

def exit_program():
    window.destroy()

def generate_response(input_text):
    # Sentiment Analysis
    sentiment_score = analyze_sentiment(input_text)
    if sentiment_score >= 0.2:
        sentiment_label = "positive"
    elif sentiment_score <= -0.2:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Named Entity Recognition
    named_entities = extract_named_entities(input_text)

    # Part-of-Speech Tagging
    pos_tags = perform_pos_tagging(input_text)

    # Add your custom logic here to generate a response based on the analyzed input
    response = "Sentiment: {} | Named Entities: {} | POS Tags: {}".format(sentiment_label, named_entities, pos_tags)
    return response

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    compound_score = sentiment_scores["compound"]
    return compound_score

def extract_named_entities(text):
    named_entities = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tagged_tokens = pos_tag(word_tokenize(sentence))
        tree = ne_chunk(tagged_tokens)
        for subtree in tree.subtrees():
            if subtree.label() != "S":
                entity = " ".join([token[0] for token in subtree.leaves()])
                named_entities.append(entity)
    return named_entities

def perform_pos_tagging(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords.words("english")]
    tagged_tokens = pos_tag(filtered_tokens)
    pos_tags = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    return pos_tags

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"
    elif tag.startswith("V"):
        return "v"
    elif tag.startswith("N"):
        return "n"
    elif tag.startswith("R"):
        return "r"
    else:
        return "n"

def display_response(response):
    output.config(state=tk.NORMAL)
    output.insert(tk.END, "Bot: " + response + "\n")
    output.config(state=tk.DISABLED)
    scroll_to_bottom()

def scroll_to_bottom(*args):
    output.yview_moveto(1.0)

nltk.download('words')

# Create the main window
window = tk.Tk()
window.title("Chatbot App")
window.configure(bg="black")

# Create the input textbox
entry = tk.Entry(window, width=50, bg="green", fg="white")
entry.pack(pady=10)

# Create the "Send" button
button = tk.Button(window, text="Send", command=process_input, bg="green", fg="white")
button.pack(pady=5)

# Create the output display area
output = tk.Text(window, height=10, width=50, bg="green", fg="white")
output.config(state=tk.DISABLED)
output.pack(pady=10)

# Configure the output Text widget to call scroll_to_bottom() whenever new text is inserted
output.config(yscrollcommand=scroll_to_bottom)

# Start the main event loop
window.mainloop()