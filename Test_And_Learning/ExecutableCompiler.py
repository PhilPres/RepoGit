import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from collections import Counter

def record_basic_words(text):
    word_list = words.words()
    tokens = word_tokenize(text)
    basic_words = [word.lower() for word in tokens if word.lower() in word_list]
    word_counts = Counter(basic_words)
    return word_counts

def classify_words_by_occurrence(word_counts):
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    words = [word for word, _ in sorted_words]
    return " ".join(words)

def main():
    nltk.download('words')
    print("Enter a sentence or a series of words:")
    user_input = input("> ")
    word_counts = record_basic_words(user_input)
    words_with_occurrence = classify_words_by_occurrence(word_counts)
    print("Words with occurrence (in descending order):")
    print(words_with_occurrence)

if __name__ == '__main__':
    main()