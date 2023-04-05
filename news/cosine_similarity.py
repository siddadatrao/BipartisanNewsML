from sklearn.feature_extraction.text import CountVectorizer


def cosine_similarity(text1, text2):
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(text1)
    vector_matrix



t1 = "hello this is a ball"
t2 = "this is not a ball"

print(cosine_similarity(t1, t2))


