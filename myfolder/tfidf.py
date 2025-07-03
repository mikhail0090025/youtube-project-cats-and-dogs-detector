from collections import defaultdict
import os
import math
import numpy as np

# Список документов
all_documents = os.listdir()
all_documents.sort()
all_documents = all_documents[:100]
print(all_documents)
all_doc_words = []
extra_symbols = ".,[](){}:;'\"/\\!?-_"

# Список всех уникальных слов
all_words = []
idf_dict = {}  # Словарь для IDF

def text_to_list(text):
    res = text.lower()
    for char in extra_symbols:
        res = res.replace(char, ' ')
    return [word for word in res.split() if word]

# Сбор слов и вычисление IDF
for document in all_documents:
    if not os.path.exists(document):
        print(f"Файл {document} не найден! Пропускаем...")
        continue
    words_in_doc = defaultdict(int)
    with open(document, "r", encoding='utf-8') as file:
        text = file.read()
        all_words_in_text = text_to_list(text)

        for word in all_words_in_text:
            if word not in all_words:
                all_words.append(word)
            words_in_doc[word] += 1
    
    all_doc_words.append(dict(words_in_doc))

# Вычисление IDF
N = len(all_documents)
for word in all_words:
    documents_with_word = sum(1 for doc_words in all_doc_words if word in doc_words)
    idf = math.log10(N / min(N, documents_with_word + 1))
    if idf < 0:
        print("N: ", N, ", documents_with_word", min(N, documents_with_word + 1), ", LOG: ", idf)
    idf_dict[word] = idf

print("Уникальные слова:", len(all_words))
print("IDF словарь (первые 5):", {k: v for k, v in list(idf_dict.items())[:5]})

# Функция для построения матрицы TF-IDF
def get_embeddings():
    matrix = np.zeros((len(all_words), len(all_documents)))
    for i, word in enumerate(all_words):
        for j, doc_words in enumerate(all_doc_words):
            total_words = sum(doc_words.values())
            tf = doc_words.get(word, 0) / total_words if total_words > 0 else 0
            idf = idf_dict.get(word, math.log10(N))  # Значение по умолчанию для новых слов
            matrix[i, j] = tf * idf
    return matrix

# Построение матрицы
embeddings = get_embeddings()
print(embeddings)
print(embeddings[:10])
print(embeddings.shape)

# Метод для предложения (для проверки)
def calc_tf_idf(text):
    words = text_to_list(text)
    tf_idfs = []
    for word in words:
        tf = words.count(word) / len(words)
        idf = idf_dict.get(word, math.log10(N))
        tf_idfs.append(tf * idf)
    return tf_idfs

# Тест
test_sentence = "I like cats dogs"
tfidf_result = calc_tf_idf(test_sentence)
print(f"Предложение: {test_sentence}")
print(f"TF-IDF веса: {tfidf_result}")