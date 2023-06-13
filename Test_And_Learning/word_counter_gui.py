import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words
from collections import Counter
import tkinter as tk

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
    word_counts = record_basic_words(user_input)
    words_with_occurrence = classify_words_by_occurrence(word_counts)
    output.config(state=tk.NORMAL)
    output.delete("1.0", tk.END)
    output.insert(tk.END, words_with_occurrence)
    output.config(state=tk.DISABLED)

nltk.download('words')

# Create the main window
window = tk.Tk()
window.title("Word Counter App")
window.configure(bg="black")

# Create the input textbox
entry = tk.Entry(window, width=50, bg="green", fg="white")
entry.pack(pady=10)

# Create the "Count Words" button
button = tk.Button(window, text="Count Words", command=process_input, bg="green", fg="white")
button.pack(pady=5)

# Create the output display area
output = tk.Text(window, height=10, width=50, bg="green", fg="white")
output.config(state=tk.DISABLED)
output.pack(pady=10)

# Start the main event loop
window.mainloop()