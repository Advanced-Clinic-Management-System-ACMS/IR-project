import numpy as np

def fuse(bm25_scores, emb_scores, alpha=0.5):

    bm25_scores = np.array(bm25_scores)
    emb_scores = np.array(emb_scores)

    return alpha * bm25_scores + (1 - alpha) * emb_scores