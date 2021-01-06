import streamlit as st
import functions

def options():
    features = {}
    
    model = st.selectbox("Choose summarization techniqe", ("Gensim summarize",  "LSA", "Text Rank", "NLTK sent_tonekizer"))
    features["model"] = model
    
    if model == "Gensim summarize":   
        gensim_per = st.number_input("Proportion of the original text", value=0.1, min_value=0.01, max_value=0.99, step=0.05)
        features["gensim_per"] = gensim_per
        
    if model == "LSA":
        lsa_num_sen = st.number_input("Number of sentences", value=5, min_value=1, max_value=functions.nltk_max_sentences(document), step=1)
        lsa_num_top = st.number_input("Number of topics (1 to 5 is recommended)", value=2, min_value=1, max_value=lsa_num_sen, step=1)
        features["lsa_num_sen"] = lsa_num_sen
        features["lsa_num_top"] = lsa_num_top
        
    if model == "Text Rank":
        text_num_sen = st.number_input("Number of sentences", value=5, min_value=1, max_value=functions.nltk_max_sentences(document), step=1)
        features["text_num_sen"] = text_num_sen
        
    if model == "NLTK sent_tonekizer":
        nltk_num_sen = st.number_input("Number of sentences", value=5, min_value=1, max_value=functions.nltk_max_sentences(document), step=1)
        features["nltk_num_sen"] = nltk_num_sen

    return features


def side_bar():
    st.sidebar.title("Model description")
    st.sidebar.write("For this project I use 4 different models. All of them are extractive summarization techniques as they are well stablished and eassy to implement.")
    st.sidebar.write("**Gensim summarize**")
    st.sidebar.write("Uses a variation of Text Rank algorithm to return the most representative sentences in a given text.")
    st.sidebar.write("**LSA**")
    st.sidebar.write("Analyze relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms.")
    st.sidebar.write("**Text Rank**")
    st.sidebar.write("A graph based method that applies the ideas of the ranking algorithm used in Google (PageRank).")
    st.sidebar.write("**NLTK sent_tonekizer**")
    st.sidebar.write("Return a sentence-tokenized copy of the text, using NLTK's recommended sentence tokenizer. May not make sense as some words are omitted. ")


st.set_page_config(
    page_title="Text Summarization",
    layout="centered",
    initial_sidebar_state="expanded")

st.subheader("Jose Villamor")

html_temp = """
    <div style="background:#eb4242 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Text Summarization </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html = True)
     
#st.image("summaries.png", use_column_width=True)  
     
st.write("This web app summarizes a given text using 4 different NLP extractive techniques. Simply insert a text, select a model and click the bottom.") 
side_bar()

st.write("**Insert text**")
document = st.text_area("5000 Max Characters ", height=400, max_chars=5000)

st.write("**Select Model and number of sentences**")
parameters = options()

st.write("**Click the button**")
if st.button("SUMMARY"):
    st.write("**Summary**")
    if parameters["model"] == "Gensim summarize":
        gensim_summary = functions.gen_summarize(document, parameters["gensim_per"])
        st.write(gensim_summary)
        
    if parameters["model"] == "LSA":
        LSA_summary = functions.LSA_sum(document, parameters["lsa_num_sen"], parameters["lsa_num_top"])
        st.write(LSA_summary)
    
    if parameters["model"] == "Text Rank":
        Text_Rank_summary = functions.text_rank_summ(document, parameters["text_num_sen"])
        st.write(Text_Rank_summary)
            
    if parameters["model"] == "NLTK sent_tonekizer":
        summary = functions.nltk_sum(document, parameters["nltk_num_sen"])
        for x in summary:
            st.write(x)

st.write("**If you want to know more about the project or others that i have done visit my github account: https://jose-villamor.github.io/Portfolio_website/portfolio.html**")

