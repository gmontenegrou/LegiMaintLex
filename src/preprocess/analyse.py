import pandas as pd
from collections import Counter

_base_path = "src/data/"
csv_path = f"{_base_path}fullcorpus_document_level.csv"

top_n = 100
df = pd.read_csv(csv_path)


# --- Token counts ---
def count_tokens(column):
    counter = Counter()
    for entry in df[column].dropna():
        # Separar por coma y limpiar espacios
        tokens = [t.strip() for t in entry.split(",") if t.strip()]
        counter.update(tokens)
    return counter


# Count verbs and nouns
verbs_counter = count_tokens("verbs")
nouns_counter = count_tokens("nouns")

# --- Show top  ---
print("Top", top_n, "verbs:")
for verb, freq in verbs_counter.most_common(top_n):
    print(f"{verb}: {freq}")

print("Top", top_n, "nouns:")
for noun, freq in nouns_counter.most_common(top_n):
    print(f"{noun}: {freq}")

# --- Optional: save ---
pd.DataFrame(verbs_counter.most_common(), columns=["verb", "count"]).to_csv(
    "verbs_count.csv", index=False
)
pd.DataFrame(nouns_counter.most_common(), columns=["noun", "count"]).to_csv(
    "nouns_count.csv", index=False
)

print("\nCounting exported to verbs_count.csv and nouns_count.csv")
