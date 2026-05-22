

# install gensim first

from gensim.models import Word2Vec

# Toy corpus (list of tokenized sentences)
sentences = [
    ["king", "queen", "man", "woman"],
    ["paris", "france", "rome", "italy"],
    ["king", "man"],
    ["queen", "woman"],
    ["paris", "capital", "france"],
    ["rome", "capital", "italy"]
]

# Train Skip-Gram model
model = Word2Vec(
    sentences,
    vector_size=50,
    window=2,
    min_count=1,
    # sg=1 -> Skip-Gram (sg=0 -> CBOW)
    sg=1,           
    epochs=200
)

# ‘’’
# Choose a sentence.
# sentence=”OOOOOOO…”
# For each word in the sentence, 
# 1) show the embedding vector of the word
# 2) show the most similar word by using model.wv.most_similar
# ‘’’
