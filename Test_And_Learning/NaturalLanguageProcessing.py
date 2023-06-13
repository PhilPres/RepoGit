from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from collections import Counter

#second version
def record_basic_words(text):
    word_list = words.words()
    tokens = word_tokenize(text)
    basic_words = [word.lower() for word in tokens if word.lower() in word_list]
    word_counts = Counter(basic_words)
    return word_counts

#first version
def record_basic_words1(text):
    tokens = word_tokenize(text)
    basic_words = [word.lower() for word in tokens if word.lower() not in stopwords.words('english')]
    word_counts = Counter(basic_words)
    return word_counts

#third version
def classify_words_by_occurrence(word_counts):
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    words = [word for word, _ in sorted_words]
    return " ".join(words)

#second version
def classify_words_by_occurrence2(word_counts):
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    words_with_occurrence = [f"{word} ({count})" for word, count in sorted_words]
    return ", ".join(words_with_occurrence)
#first version
def classify_words_by_occurrence_1(word_counts):
    num_occurrences = word_counts.values()
    max_occurrence = max(num_occurrences)
    min_occurrence = min(num_occurrences)
    range_occurrence = max_occurrence - min_occurrence

    thresholds = [
        min_occurrence + range_occurrence // 3,
        min_occurrence + (2 * range_occurrence) // 3
    ]

    word_categories = {
        "High Frequency": [],
        "Medium Frequency": [],
        "Low Frequency": []
    }

    for word, count in word_counts.items():
        if count >= thresholds[1]:
            word_categories["High Frequency"].append((word, count))
        elif count >= thresholds[0]:
            word_categories["Medium Frequency"].append((word, count))
        else:
            word_categories["Low Frequency"].append((word, count))

    return word_categories

def main():
    text = "ball it is to ball like sample"
    word_counts = record_basic_words(text)
    words_with_occurrence = classify_words_by_occurrence(word_counts)
    print(words_with_occurrence)


    #word_categories = classify_words_by_occurrence(word_counts)

    #for category, words in word_categories.items():
    #    print(f"{category}:")
    #   for word, count in words:
    #        print(f"{word}: {count}")
    #    print()

if __name__ == '__main__':
    main()