import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Load the dataset
file_path = './top_forwarded_messages.xlsx'
df = pd.read_excel(file_path)

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stop_words = {'ukraine', 'ukrainian', 'ukrainians', 'russia', 'russian', 'russians'}

# Update stopwords with custom stopwords
stop_words.update(custom_stop_words)

# Function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

# Function to preprocess text
def preprocess_text(text):
    text = remove_urls(text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove digits
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Preprocess the data
df['cleaned_translation'] = df['translation'].astype(str).apply(preprocess_text)

# Use CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
X = vectorizer.fit_transform(df['cleaned_translation'])

# Convert processed text to list of tokens for gensim
texts = df['cleaned_translation'].apply(lambda x: x.split()).tolist()

# Create a dictionary representation of the documents
dictionary = Dictionary(texts)

# Create a corpus: list of bags of words
corpus = [dictionary.doc2bow(text) for text in texts]

# Function to compute coherence score
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        model.fit(X)
        model_list.append(model)
        # Generate topics for gensim coherence model
        topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-51:-1]] for topic in model.components_]
        coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values

# Determine the optimal number of topics
limit = 10  # Max number of topics is 10
start = 5   # Min number of topics is 5
step = 1

model_list, coherence_values = compute_coherence_values(dictionary, corpus, texts, limit, start, step)

# Plot coherence values to find the optimal number of topics
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Coherence Score by Number of Topics")
plt.show()

# Select the model with the highest coherence score
optimal_model = model_list[np.argmax(coherence_values)]
optimal_num_topics = start + np.argmax(coherence_values) * step

print(f"Optimal Number of Topics: {optimal_num_topics}")

# Function to display topics
def display_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(topic_words)
    return topics

# Display the topics
n_top_words = 50
feature_names = vectorizer.get_feature_names_out()
topics = display_topics(optimal_model, feature_names, n_top_words)

# Compute topic distribution for each document
doc_topic_dist = optimal_model.transform(X)

# Assign each document to the most relevant topic
dominant_topic = np.argmax(doc_topic_dist, axis=1)

# Calculate the proportion of documents in each topic
topic_sizes = np.bincount(dominant_topic)
topic_proportions = topic_sizes / len(dominant_topic)

# Create a DataFrame to store topics and their sizes
topics_df = pd.DataFrame({
    'Topic': range(1, len(topic_sizes) + 1),
    'Size': topic_sizes,
    'Proportion': topic_proportions,
    'Keywords': topics
}).sort_values(by='Size', ascending=False).reset_index(drop=True)

# Function to compute cosine similarity between texts
def text_cosine_similarity(text1, text2, vectorizer):
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# Function to get representative examples for each topic with diversity
def get_representative_examples(model, X, df, vectorizer, n_examples=3, similarity_threshold=0.5):
    doc_topic_dist = model.transform(X)
    top_texts = {}
    for topic_idx in range(model.n_components):
        top_text_indices = np.argsort(doc_topic_dist[:, topic_idx])[::-1]
        selected_texts = []
        seen_texts = set()
        for idx in top_text_indices:
            text = df.iloc[idx]['translation']
            sender = df.iloc[idx]['sender']
            url = df.iloc[idx]['url']
            date = df.iloc[idx]['date']
            if text not in seen_texts:
                # Check if the text is sufficiently distinct from already selected examples
                is_distinct = all(text_cosine_similarity(text, seen_text, vectorizer) < similarity_threshold for seen_text in seen_texts)
                if is_distinct:
                    selected_texts.append({
                        'text': text[:100],  # Truncate to the first 100 characters
                        'sender': sender,
                        'url': url,
                        'date': date
                    })
                    seen_texts.add(text)
                if len(selected_texts) >= n_examples:
                    break
        top_texts[topic_idx] = selected_texts
    return top_texts

# Get representative examples for each topic
top_texts = get_representative_examples(optimal_model, X, df, vectorizer)

# Display the topics with examples
for index, row in topics_df.iterrows():
    topic_idx = row['Topic'] - 1
    print(f"Topic #{topic_idx + 1} (Size: {row['Size']}, Proportion: {row['Proportion'] * 100:.2f}%):")
    print(f"Keywords: {', '.join(row['Keywords'])}")
    print("Examples:")
    for example in top_texts[topic_idx]:
        print(f" - Text: {example['text']}")
        print(f"   Sender: {example['sender']}")
        print(f"   URL: {example['url']}")
        print(f"   Date: {example['date']}")
    print()
