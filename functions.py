import re
from gensim.summarization import summarize
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import networkx

#FEATURE ENGENEERING FOR LSA AND TEXT RANK
def feature_engineering(doc):
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(nltk_sum(doc, feature_engi=True))
    dt_matrix = dt_matrix.toarray()
    td_matrix = dt_matrix.T
    return td_matrix

#GENSIM
def gen_preprocess(doc):
    doc_pre = re.sub(r'\n|\r', '', doc)
    doc_pre = re.sub(r' +', '', doc)
    doc_pre = re.sub(',(?!\s+\d$)', '', doc)
    doc_pre = doc_pre.strip()
    return doc_pre
    
def gen_summarize(doc, ratio=0.2):
    return summarize(gen_preprocess(doc), ratio=ratio, split=False)


#NLTK
def normalize_document(doc):
    stop_words = nltk.corpus.stopwords.words('english')
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def nltk_sum(doc, num_sent=7, feature_engi=False):
    sentences = nltk.sent_tokenize(doc)
    normalize_corpus = np.vectorize(normalize_document)
    norm_sentences = normalize_corpus(sentences)
    
    if feature_engi == False:
        return norm_sentences[:num_sent]
    else:
        return norm_sentences
    
def nltk_max_sentences(doc):
    return len(nltk.sent_tokenize(doc))

#LSA
def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


def LSA_sum(doc, num_sentences=7, num_topics=3):
    u, s, vt = low_rank_svd(feature_engineering(doc), singular_count=num_topics)
    term_topic_mat, singular_values, topic_document_mat = u, s, vt
    
    # remove singular values below threshold                                         
    sv_threshold = 0.5
    min_sigma_value = max(singular_values) * sv_threshold
    singular_values[singular_values < min_sigma_value] = 0 
    
    salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))  
    top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
    top_sentence_indices.sort()
    
    sentences = nltk.sent_tokenize(doc)
    
    return ' '.join(np.array(sentences)[top_sentence_indices])


#TEXT RANK
def text_rank_summ(doc, num_sentences=7):
    
    similarity_matrix = np.matmul(feature_engineering(doc).T, feature_engineering(doc))
    np.round(similarity_matrix, 3)
    
    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    ranked_sentences[:num_sentences]
   
    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()
    
    sentences = nltk.sent_tokenize(doc)
    
    return ' '.join(np.array(sentences)[top_sentence_indices])
    













