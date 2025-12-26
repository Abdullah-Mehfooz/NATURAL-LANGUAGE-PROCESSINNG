
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import pandas as pd
from nltk import word_tokenize, pos_tag

text = "The quick brown fox jumps over the lazy dog"

# ===============================
# 4. TOKENIZATION
# ===============================
tokens = word_tokenize(text)

# ===============================
# 5. POS TAGGING
# ===============================
pos_tags = pos_tag(tokens)

# ===============================
# 6. DISPLAY RESULTS
# ===============================
for word, tag in pos_tags:
    print(f"{word} --> {tag}")

# ===============================
# 7. OPTIONAL: SAVE AS DATAFRAME
# ===============================
df = pd.DataFrame(pos_tags, columns=["Word", "POS_Tag"])
print(df)