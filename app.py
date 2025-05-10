import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

# Function to find similar characters based on t-SNE 2D coordinates
def find_similar_characters(character_name, df, top_n=5):
    target_row = df[df['character'] == character_name]
    
    if target_row.empty:
        return f"{character_name} not found in dataset."
    
    target_vector = target_row[['x', 'y']].values
    all_vectors = df[['x', 'y']].values
    distances = euclidean_distances(target_vector, all_vectors)[0]
    
    df['distance'] = distances
    similar_characters = df[df['character'] != character_name].sort_values(by='distance').head(top_n)
    
    return similar_characters[['character', 'distance']]

# Streamlit app
def main():
    st.set_page_config(page_title="Game of Thrones Character Match", page_icon="👑")
    st.title('Game of Thrones Character Match Prediction 👑⚔️')

    # Load dataset
    df = pd.read_json('script-bag-of-words.json')

    # Create a dictionary of dialogues per character
    dialogue = {}
    for index, row in df.iterrows():
        for item in row['text']:
            if item['name'] in dialogue:
                dialogue[item['name']] += item['text']
            else:
                dialogue[item['name']] = item['text'] + " "

    # Create DataFrame
    new_df = pd.DataFrame({
        'character': dialogue.keys(),
        'words': dialogue.values()
    })
    new_df['num_words'] = new_df['words'].apply(lambda x: len(x.split()))
    new_df = new_df.sort_values('num_words', ascending=False).head(100)

    # Embedding and t-SNE
    cv = CountVectorizer(stop_words='english')
    embeddings = cv.fit_transform(new_df['words']).toarray().astype('float64')
    tsne = TSNE(n_components=2, random_state=123)
    z = tsne.fit_transform(embeddings)

    new_df['x'] = z.T[0]
    new_df['y'] = z.T[1]

    # Character selection
    character_list = new_df['character'].tolist()
    selected_character = st.selectbox("Select a Character ⚔️", character_list)

    if st.button('Match Character 🔍'):
        st.subheader(f"Characters similar to **{selected_character}**:")
        similar_characters = find_similar_characters(selected_character, new_df, top_n=5)
        st.dataframe(similar_characters)

        # Plot t-SNE
        fig = px.scatter(new_df, x='x', y='y', color='character',
                         title="Character Embedding (t-SNE Visualization)")
        st.plotly_chart(fig)

# Run the app
if __name__ == '__main__':
    main()
