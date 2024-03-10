import inflect
from collections import Counter
from operator import itemgetter
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pickle
from google.cloud import storage
import math
from nltk.stem import PorterStemmer
import nltk

nltk.download('wordnet')
import hashlib


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


bucket_name = '211458294_tiltan'
full_path = f"gs://{bucket_name}/"
paths = []
client = storage.Client()
blobs = client.list_blobs(bucket_name)
for b in blobs:
    if b.name[-7:] == "parquet":
        paths.append(full_path + b.name)

# define stop words and regex for preprocessing of query
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](["'\-]?\w){2,24}""", re.UNICODE)
RE_NUMBER = re.compile(r'\b\w*\d+\w*\b')


# following methods are for loading several indexes and dictionaries from bucket
def load_doc_id_title_dict():
    file_path_in_bucket = "doc_id_title_dict.pickle"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    loaded_dict = pickle.loads(downloaded_pickled_content)
    return loaded_dict


def load_doc_id_page_views():
    file_path_in_bucket = "doc_id_page_views_dict.pickle"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    wid2pv = pickle.loads(downloaded_pickled_content)
    return wid2pv


def load_doc_id_pr():
    file_path_in_bucket = "doc_id_pagerank_dict.pickle"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    pagerank_dict = pickle.loads(downloaded_pickled_content)
    return pagerank_dict


def load_title_index():
    file_path_in_bucket = 'title/title_index.pkl'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    title_inverted_index = pickle.loads(downloaded_pickled_content)
    return title_inverted_index


def load_body_index():
    file_path_in_bucket = 'body/body_index.pkl'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    body_inverted_index = pickle.loads(downloaded_pickled_content)
    return body_inverted_index


def load_title_docs_wrong_vector_sizes_dict():
    file_path_in_bucket = "title_docs_wrong_vector_sizes_dict.pickle"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    loaded_dict = pickle.loads(downloaded_pickled_content)
    return loaded_dict


def load_body_docs_wrong_vector_sizes_dict():
    file_path_in_bucket = "body_docs_wrong_vector_sizes_dict.pickle"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path_in_bucket)
    # Download the pickle file from Google Cloud Storage
    downloaded_pickled_content = blob.download_as_bytes()
    loaded_dict = pickle.loads(downloaded_pickled_content)
    return loaded_dict


# putting it all together
def load_everything():
    doc_title_dict = load_doc_id_title_dict()
    doc_pageviews_dict = load_doc_id_page_views()
    doc_pagerank_dict = load_doc_id_pr()
    title_inverted_index = load_title_index()
    body_inverted_index = load_body_index()
    title_docs_wrong_vector_sizes_dict = load_title_docs_wrong_vector_sizes_dict()
    body_docs_wrong_vector_sizes_dict = load_body_docs_wrong_vector_sizes_dict()
    return doc_title_dict, doc_pageviews_dict, doc_pagerank_dict, title_inverted_index, body_inverted_index, title_docs_wrong_vector_sizes_dict, body_docs_wrong_vector_sizes_dict


doc_title_dict, doc_pageviews_dict, doc_pagerank_dict, title_inverted_index, body_inverted_index, title_docs_wrong_vector_sizes_dict, body_docs_wrong_vector_sizes_dict = load_everything()


# functionality for preprocessing the query (similar to how we preprocessed docs before indexing)
def preprocess_text(text):

    # Function to expand numbers in the query (converting numbers to their literal representation)
    def expand_numbers(tokens):
        # Create an inflect engine
        p = inflect.engine()
        query = " ".join(tokens)
        # Find all numbers in the query
        numbers = RE_NUMBER.findall(query)

        # Expand each number to its word representation
        for num in numbers:
            word_representation = p.number_to_words(num)
            tokens.append(word_representation)
        return tokens

    def remove_stopwords(tokens):
        filtered_tokens = [token for token in tokens if token not in all_stopwords]
        return filtered_tokens

    def stemming(tokens):
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    tokens = text.split(" ")
    tokens = expand_numbers(tokens)
    text = " ".join(tokens)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens


def cos_similarity_title(base_dir, bucket_name, query, inverted):
    similarity_dict = {}    # dict containing doc id and its similarity to the query
    words_list = query.split(" ")
    word_counts = Counter(words_list)
    words_dict = dict(word_counts)
    for word in words_dict:
        current_posting = inverted.read_a_posting_list(base_dir, word, bucket_name)
        for doc_id, tf_idf in current_posting:
            updated_similarity = similarity_dict.get(doc_id, 0) + tf_idf * words_dict[word]
            similarity_dict[doc_id] = updated_similarity
    for doc_id in similarity_dict.keys():
        similarity_dict[doc_id] = similarity_dict[doc_id] * (1 / len(words_list))
        similarity_dict[doc_id] = similarity_dict[doc_id] * title_docs_wrong_vector_sizes_dict[doc_id]
    return similarity_dict


def cos_similarity(base_dir, bucket_name, query, inverted, threshold):
    similarity_dict = {}    # dict containing doc id and its similarity to the query
    count_query_words_in_docs = {}
    words_list = query.split(" ")
    word_counts = Counter(words_list)
    words_dict = dict(word_counts)
    for word in words_dict:
        current_posting = inverted.read_a_posting_list(base_dir, word, bucket_name)
        for doc_id, tf_idf in current_posting:
            count_query_words_in_docs[doc_id] = count_query_words_in_docs.get(doc_id, 0) + 1
            updated_similarity = similarity_dict.get(doc_id, 0) + tf_idf * words_dict[word]
            similarity_dict[doc_id] = updated_similarity
    for doc_id in similarity_dict.keys():
        similarity_dict[doc_id] = similarity_dict[doc_id] * (1 / len(words_list))
    # filter docs that don't include all query words, boolean model
    filtered_dict = {doc_id: value for doc_id, value in count_query_words_in_docs.items() if value >= threshold}
    intersection_keys = set(similarity_dict.keys()) & set(filtered_dict.keys())
    result_dict = {key: similarity_dict[key] for key in intersection_keys}
    return result_dict


# calculate combined ranks for docs
def combine_dicts(title_sim_dict, body_sim_dict, doc_pagerank_dict, doc_pageviews_dict):
    combined_rank_dict = {}
    for doc_id in set(title_sim_dict.keys() | body_sim_dict.keys()):
        title_weight = 0.6
        body_weight = 0.4
        if doc_id not in title_sim_dict.keys() and doc_id in body_sim_dict.keys():
            title_weight = 0.4
            body_weight = 0.6
        title_sim = title_sim_dict.get(doc_id, 0)
        body_sim = body_sim_dict.get(doc_id, 0)
        pagerank = doc_pagerank_dict.get(doc_id, 0)
        pageviews = doc_pageviews_dict.get(doc_id, 0)
        overall_rank = title_weight * title_sim + body_weight * body_sim + math.log(1 + pagerank, 1500) + math.log(1 + pageviews, 1500)
        combined_rank_dict[doc_id] = overall_rank
    return combined_rank_dict


# return 100 must relevant docs for query
def get_relevant_docs(query, title_inverted_index, body_inverted_index, doc_title_dict, doc_pagerank_dict,
                      doc_pageviews_dict):
    title_sim_dict = cos_similarity_title('.', '211458294_tiltan', query, title_inverted_index)
    body_sim_dict = cos_similarity('.', '211458294_tiltan', query, body_inverted_index, len(query.split(" ")))
    combined_rank_dict = combine_dicts(title_sim_dict, body_sim_dict, doc_pagerank_dict, doc_pageviews_dict)
    sorted_ranked_docs = sorted(combined_rank_dict.items(), key=itemgetter(1), reverse=True)[:100]
    result_list = [(str(doc_id), doc_title_dict[doc_id]) for doc_id, _ in sorted_ranked_docs]
    return result_list


# return 100 must relevant docs for query
def search_in_backend(query):
    query = " ".join(preprocess_text(query))
    return get_relevant_docs(query, title_inverted_index, body_inverted_index, doc_title_dict, doc_pagerank_dict,
                             doc_pageviews_dict)
