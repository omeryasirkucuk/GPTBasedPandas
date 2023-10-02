import streamlit as st
import pandas as pd
import openai
import os
from openai.embeddings_utils import get_embedding,cosine_similarity
import uuid
import json
import numpy as np
import random


# initialize emoji as a Session State variable
if "emoji" not in st.session_state:
    st.session_state.emoji = "👈"





st.set_page_config(page_title="Ask your PDL Data 🦊", initial_sidebar_state="collapsed")

st.markdown("<h1 style='text-align: center; color: grey;'>Ask Your PDL Data 🐼</h1>", unsafe_allow_html=True)





def manipulate_data(data):
    
    def duzelt(text):
        # Türkçe karakterleri düzelt
        text = text.replace('I', 'i').replace('İ', 'i').title()
        return text

    data['Feature Tanim'] = data['Feature Tanim'].apply(duzelt)
    data['Market Tanim'] = data['Market Tanim'].apply(duzelt)
    data['Entity Tanim'] = data['Entity Tanim'].apply(duzelt)

    df = data.copy()
    df = df.drop(["Paket Tanimi", "SVO", "Paket No"], axis=1)

    df.loc[df["Kullanim S/O"] == "O", "Feature Option Situation"] = "Optional"
    df.loc[df["Kullanim S/O"] == "S", "Feature Option Situation"] = "Standart"

    df.loc[df["Model"] == "H625", "Vehicle Full Name"] = "F-MAX or FMAX or Tractor or 625 or H625"
    df.loc[df["Model"] == "H566", "Vehicle Full Name"] = "Legacy or Construction  or İnşaat or Legacy Tractor or 566 or H566"

    df['Vehicle Full Name'] = df.apply(
        lambda row: row['Vehicle Full Name'] + ' or Lowliner' if 'LL' in row['Entity Tanim'] else row[
            'Vehicle Full Name'], axis=1)

    df['Transmission Box'] = df['Entity Tanim'].apply(lambda x: 'Ecotorq' if 'Ecotorq' in x else "Other")

    df['Euro Standart'] = df['Entity Tanim'].apply(lambda x: 'Euro6' if 'E6' in x else 'Euro5')

    df['Transmission Count'] = df['Entity Tanim'].apply(
        lambda x: '16 gears or 16 forward' if '16' in x else ('12 gears or 12 forward' if '12' in x else 'Others'))

    df = df.rename(columns={'Market': 'Market Code', 'Model Tanim': 'Model Detail', 'Market Tanim': 'Country / Market',
                            'Entity Tanim': 'Entity Detail', 'Feature Tanim': 'Feature Detail'})

    df = df.drop(["Kullanim S/O"], axis=1)
    df = df.drop(["Model"], axis=1)
    df = df.drop(["Market Code"], axis=1)
    df = df.drop(["Feature"], axis=1)
    df = df.drop(["Entity"], axis=1)

    df2 = pd.DataFrame()
    df2["Combined"] = df.apply(lambda row: ', '.join([f"{col}: {val}" for col, val in zip(df.columns, row)]), axis=1)
    return df2

def learn_dataframe(df):
    # OpenAI API anahtarını ayarla
    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key = os.environ['OPENAI_API_KEY']
    content_chunks = []

    for index, row in df.iterrows():
        content = row["Combined"]
        obj = {
            "id": str(uuid.uuid4()),
            "text": content,
            "embedding": get_embedding(content, engine='text-embedding-ada-002')
        }
        content_chunks.append(obj)

    # Save the learned data into the knowledge base
    json_file_path = 'my_knowledgebase.json'

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.extend(content_chunks)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def Answer_from_documents(user_query, learned_data):
    user_query_vector = get_embedding(user_query, engine='text-embedding-ada-002')

    for item in learned_data:
        item['embeddings'] = [np.array(embedding).reshape(1, -1) for embedding in item['embedding']]

    for item in learned_data:
        item_embedding = np.concatenate(item['embeddings'], axis=1)
        item['similarities'] = cosine_similarity(item['embedding'], user_query_vector)
    sorted_data = sorted(learned_data, key=lambda x: x['similarities'], reverse=True)

    context = ''
    for item in sorted_data[:2]:
        context += item['text']

    myMessages = [
        {"role": "system", "content": "You're a helpful Assistant."},
        {"role": "user", "content": "The following is a Context:\n{}\n\n Answer the following user query according to the above given context.\n\nquery: {}".format(context, user_query)}
    ]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=myMessages,
        max_tokens=200,
    )

    return response['choices'][0]['message']['content']

# CSV dosyasını yükle
csv_file = st.file_uploader("Load Your CSV 👈", type="csv")



# Dosya yüklendiyse
if csv_file is not None:
    # Pandas dataframe'e çevir
    df = pd.read_csv(csv_file, sep=";")

    # OpenAI API anahtarını ayarla
    api_key_input = st.text_input("Write your OpenAI API")
    os.environ["OPENAI_API_KEY"] = api_key_input
    openai.api_key = os.environ['OPENAI_API_KEY']


    # Soruyu al
    user_query = st.text_input("Ask the PDL 🤖")

    # Veriyi mani"püle et
    manipulated_data = manipulate_data(df)

    # Öğrenilen veriyi kontrol et
    json_file_path = 'my_knowledgebase.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            learned_data = json.load(f)
    except FileNotFoundError:
        learned_data = []

    # Manipüle edilen veriyi göster
    st.write("Manipulated Data 🛠️")
    st.dataframe(manipulated_data, width=10000)

    # Soruya cevap al
    if st.button("Submit"):
        answer = Answer_from_documents(user_query, learned_data)
        st.write("Answer:", answer)

