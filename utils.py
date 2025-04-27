import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from plotly.colors import qualitative
import streamlit as st
from gensim.parsing.preprocessing import remove_stopwords

# 訓練 Word2Vec 模型（skip-gram or CBOW）
def train_word2vec(sentences, vector_size=200, window=10, min_count=1, workers=4, sg=1):
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    model = Word2Vec(
        tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg  # sg=1 是 skip-gram，sg=0 是 CBOW
    )
    return model

# 計算句子向量（平均 word vectors）
def sentence_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 預處理：去 stopwords + token 化
def preprocess_sentence(text):
    return simple_preprocess(remove_stopwords(text))


def get_color_list(num_sentences):
    base_colors = qualitative.Plotly
    return [base_colors[i % len(base_colors)] for i in range(num_sentences)]

def plot_word2vec_2d(text):
    if not text.strip():
        st.warning("請輸入至少一句話，每一句請換行。")
        return

    sentences = text.strip().split("\n")  # 一句一行
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

    # 確保每一句都有詞
    tokenized_sentences = [s for s in tokenized_sentences if len(s) > 0]
    if len(tokenized_sentences) == 0:
        st.error("輸入的句子都沒有可用的單詞，請重新輸入。")
        return

    # Train Word2Vec
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

    # PCA 降維
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(word_vectors)

    # 自動產生顏色
    color_list = get_color_list(len(tokenized_sentences))

    # 配每個詞的顏色
    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_list[i])
                break

    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

    # 畫點
    scatter = go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=8),
        customdata=word_colors,
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    # 畫線（句子連接）
    display_array = [True] * len(tokenized_sentences)  # 全部句子顯示
    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        if display_array[i]:
            line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv.key_to_index]
            if len(line_vectors) >= 2:  # 至少兩個點才畫線
                line_trace = go.Scatter(
                    x=[vector[0] for vector in line_vectors],
                    y=[vector[1] for vector in line_vectors],
                    mode='lines',
                    line=dict(color=color_list[i], width=1, dash='solid'),
                    showlegend=True,
                    name=f"Sentence {i+1}"
                )
                line_traces.append(line_trace)

    # 合併圖表
    fig = go.Figure(data=[scatter] + line_traces)
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Visualization of Word Embeddings",
        width=800,
        height=800
    )

    # 顯示圖表
    st.plotly_chart(fig)

# 3D 視覺化 function
def plot_word2vec_3d(text):
    if not text.strip():
        st.warning("請輸入至少一句話，每一句請換行。")
        return

    sentences = text.strip().split("\n")
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    tokenized_sentences = [s for s in tokenized_sentences if len(s) > 0]
    if len(tokenized_sentences) == 0:
        st.error("輸入的句子都沒有可用的單詞，請重新輸入。")
        return

    # Train Word2Vec
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

    # PCA 降到 3 維
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(word_vectors)

    # 產生顏色
    color_list = get_color_list(len(tokenized_sentences))
    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_list[i])
                break

    # 3D scatter plot
    scatter = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=4)
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        title="3D Visualization of Word Embeddings",
        width=800,
        height=800
    )

    st.plotly_chart(fig)
