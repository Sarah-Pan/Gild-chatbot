import streamlit as st
from utils import train_word2vec, sentence_vector, preprocess_sentence


st.title("CBOW Model - Test Your Own Sentence")

# 訓練資料
sentences = [
    'When predictions support decisions they may influence the outcome they aim to predict.',
    'We thus also give the first sufficient conditions for retraining to overcome strategic feedback effects.',
    'Consider a simplified example of predicting credit default risk.',
    'Once recognized, performativity turns out to be ubiquitous.',
    'This raises fundamental questions.',
    'Rather, we rely on an assumption about the sensitivity of the data-generating distribution to changes in the model parameters.',
    'The latter work considers repeated risk minimization, but from the perspective of what it does to a measure of disparity between groups.',
    'In general, any instance of performative prediction can be reframed as a reinforcement learning or contextual bandit problem.',
    'We return to discuss some of the connections between both frameworks later on.',
    'We introduce the notion of performative stability to refer to predictive models that satisfy this property.'
]

# ✅ 訓練 CBOW (sg=0)
model = train_word2vec(sentences, sg=0)

# 使用者輸入測試句子
user_input = st.text_input("請輸入你想測試的句子：", "Predictions can sometimes affect decisions.")
if user_input:
    processed_sentence = preprocess_sentence(user_input)
    st.write("Tokenized Sentence:", processed_sentence)

    # 計算句子向量
    new_sentence_vec = sentence_vector(processed_sentence, model)
    st.subheader("New Sentence Vector (mean of word vectors):")
    st.write(new_sentence_vec)
    st.write(f"向量 shape: {new_sentence_vec.shape}")

    # 顯示 predictions 的相似詞
    if 'predictions' in model.wv:
        st.subheader("Most Similar Words to 'predictions':")
        similar_words = model.wv.most_similar('predictions')
        for word, score in similar_words:
            st.write(f"{word}: {score:.4f}")
    else:
        st.error("'predictions' is not in vocabulary!")
