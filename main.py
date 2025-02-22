
import json
from datetime import datetime
from pprint import pprint

import bisect
import numpy as np
import unicodedata
import re
from pathlib import Path
from collections import Counter

from firewall.functions import portInPortRange
from scipy.sparse import csr_matrix
from sympy.polys.numberfields.utilities import coeff_search

# cha = character
data_path = Path(__file__).parent / "data" / "42bin_haber" / "news" / "yasam" # yasam = life

tokenize_files = Path(__file__).parent / "TurkishTokenizer" / "lower_cha" / "BPE_32" / "tokenizer.json"

with open(tokenize_files, "r", encoding="utf-8") as file:
    bpe_tokenize_data = json.load(file)


pprint("1 - Data Loading")
general_start = datetime.now()
start = general_start
sample = []

for file_path in data_path.glob("**/*.txt"):
    with file_path.open("r", encoding="utf-8") as file:
        file_content = file.read()
        sample.append(file_content)


pprint(f"Sample size: {len(sample)} Complete Duration: {datetime.now() - general_start}")

pprint(f"1.1 - Document Exrtaction")
start = datetime.now()
sample = [doc for doc in sample]
pprint(f"Number of Documents Extracted: {len(sample)} Complete Duration: {datetime.now() - start}")

pprint("2 - Text Preprocessing")

pprint("2.1 - Character Preprocessing")
start = datetime.now()

sample = [text.lower() for text in sample]
sample = [unicodedata.normalize('NFKC', text) for text in sample]
choosen_categories = ["Ll", "Nd", "Zs"]

for i, samp in enumerate(sample):
    categories = [unicodedata.category(cha) for cha in samp]
    new_text = "".join([samp[j] if categories[j] in choosen_categories and categories[j] != 'Zs'
                        else ' ' for j in range(len(samp))])

pattern = f'[{"\\|".join(choosen_categories)}]'

for i, samp in enumerate(sample):
    new_text = re.sub(pattern, '', samp)
    new_text = re.sub(' +', ' ', new_text)
    sample[i] = new_text.strip()

pprint(f"Complete Duration: {datetime.now() - start}")

pprint("2.2 - Text Fragmentation")
start = datetime.now()

fragmented_news = []

for samp in sample:
    tokenized_new = samp.split()
    fragmented_new = [bpe_tokenize_data.get(token, token) for token in tokenized_new]
    fragmented_new.append(fragmented_new)

pprint(f"Complete Duration: {datetime.now() - start}")

pprint("3 -Numerical Encoding")

pprint("3.1 - Dictionary Creation")
start = datetime.now()
dictionary = set()

for samp in sample:
    dictionary.update(samp)

dictionary = list(dictionary)
dictionary.sort()

pprint(f"Dictionary Size: {len(dictionary)} Complete Duration: {datetime.now() - start}")

pprint(f"3.2 - Numerical Encoding")
start = datetime.now()

numerical_sample = []

for samp in sample:
    numerical_samp = [bisect.bisect_left(dictionary, word) for word in samp]
    numerical_sample.append(numerical_samp)

pprint(f"Complete Duration: {datetime.now() - start}")

pprint(f"3.2.1 - Doc Frequency Calculation")
start = datetime.now()

frequency_sample = [Counter(doc) for doc in numerical_sample]
pprint(f"Complete Duration: {datetime.now() - start}")

pprint(f"3.2.1.1 - Term Frequency Calculation")
start = datetime.now()

N = len(sample)
M = len(dictionary)

tdm = np.zeros((N, M))


for i, frequencies in enumerate(frequency_sample):
    avg_freq = sum(frequencies.values()) / len(frequencies)
    coefficient = 1 / (1 + np.log10(avg_freq))
    for j, freq in frequencies.items():
        tdm[i, j] = 1 + np.log10(1 + np.log10(coefficient))


pprint(f"Complete Duration: {datetime.now() - start}")

pprint(f"3.2.1.2 - TF -IDF Calculation")
start = datetime.now()

A = tdm > 0
df = A.sum(axis=0)
idf = np.log10(N - df/df)

tfidf = tdm * idf

pprint(f"Complete Duration: {datetime.now() - start}")

pprint(f"3.2.2 - Doc Vector Normalization")
start = datetime.now()

doc_lengths = (tfidf ** 2).sum(axis=1)

for i in range(N):
    tfidf[i, :] = tfidf[i, :] / np.sqrt(doc_lengths[i])

pprint(f"Complete Duration: {datetime.now() - start}")

pprint(f"3.3 - Sparse Matrix Conversion")
start = datetime.now()

tfidf_sparse = csr_matrix(tfidf)
d5 = tfidf_sparse[5, :]
d5 = tfidf_sparse.data[tfidf_sparse.indptr[5]:tfidf_sparse.indptr[6]]

pprint(f"General Complete Duration: {datetime.now() - general_start}")

pprint(tfidf_sparse)

no_zeros = tfidf_sparse.nnz

total_number = tfidf_sparse.shape[0] * tfidf_sparse.shape[1]

fullness_rate = no_zeros / total_number

pprint(f"Fullness Rate: {fullness_rate}")