# Game of Thrones Character Match Prediction 👑⚔️

This Streamlit app helps find similar Game of Thrones characters based on their dialogue using NLP, embeddings, and t-SNE visualization.

## 🧠 How It Works

- Uses bag-of-words embeddings from character dialogues
- Reduces dimensions with t-SNE
- Plots and compares characters based on dialogue similarity

## 📂 Dataset

Uses `script-bag-of-words.json` containing dialogue data.

## 🚀 Run the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch Streamlit:

```bash
streamlit run app.py
```

## 👤 Author

Built with ❤️ using Python, Streamlit, and scikit-learn.
