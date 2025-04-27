import streamlit as st
from utils import plot_word2vec_2d

st.title("Word2Vec 2D Visualization")
user_input = st.chat_input("請輸入句子，每一句換一行")
if user_input:
    st.write("你輸入的句子：")
    st.code(user_input)
    plot_word2vec_2d(user_input)
