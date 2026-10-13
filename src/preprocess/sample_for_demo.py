import pandas as pd
import re
import random

SELECTED_ARTICLES = {
    "R4224-12", "L125-5", "R125-5", "L111-7", "R4214-26", "R4214-28",
    "R4225-6", "R4225-8", "L111-7-1", "R111-19", "R111-19-5",
    "L111-7-3", "R111-19-7", "R111-19-12", "L111-7-5", "L111-7-12",
    "D111-19-31", "R111-19-47", "L111-7-9", "D111-19-46", "L111-7-4",
    "R111-19-27", "R123-22", "R111-19-60", "R111-18", "R111-18-7",
    "L111-7-2", "R111-18-8", "R111-18-11", "L1112-1", "L1112-3",
    "D1112-1", "D1112-7-1", "L1112-4", "D1112-8", "D1112-15",
    "L1112-2-1", "L1112-2-4", "R1112-11", "R1112-22", "L3111-7-1",
}


def article_matches(content):
    """Return the number of selected articles found in content."""
    normalized = re.sub(r"(?<=[A-Za-z])\s+(?=\d)", "", str(content))
    return sum(
        bool(re.search(rf"\b{re.escape(article)}\b", normalized, re.IGNORECASE))
        for article in SELECTED_ARTICLES
    )

df = pd.read_csv("src/data/maintreg_database_clean.csv")
df.columns

## recover art num if not existing

art_num = dict()
ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?:Art\.|Article)\s+(.{1,32}?)(?:\s*\.\s*-|\s*[.):-]|\s*$)|ANNEXE\s+(.{1,32}?.)",
    re.IGNORECASE,
)

ARTICLE_NUMBER_RE_2 = re.compile(
    r"^\s*(?:Art(?:icle)?\.?\s+(?:\d+|1er))\s*(?:\.\s*-|[.):-])?\s*",
    re.IGNORECASE,
)

for i, row in df.iterrows():
    cnt = row["content"]
    num = row["num"]
    cnt_new = re.sub(ARTICLE_NUMBER_RE_2, "", cnt)
    df.at[i, "content"] = cnt_new
    if not pd.notna(num):
        art_new = re.search(ARTICLE_NUMBER_RE, cnt[:25])
        if art_new:
            art_neww =  re.sub(r"[.\s-]+", "", art_new.group(0))
            art_num[i] = art_neww
            # pass
        else:
            # print(cnt[:25])
            pass
        # current_id = str(idR) + str(num)

## data filtered

DEFAULT_PATTERN = r"\bportes?\b|\bportails?\b|\bascenseur?\b|\bescaliers?\b"
contents = []
tokens_nb = []
currents_id = []
keywords = []
nb = []
reference_scores = []
idx = []
for i, row in df.iterrows():
    file = row["content"]
    
    topic = row["domain"]
    tokens = row["content_token_count"]
    num = row["num"]
    idR = row["id"]
    
    if "porte" in topic.lower() or "accessibilité" in topic.lower():
        nb.append(file)
        MPATTERN = re.search(DEFAULT_PATTERN, file, re.IGNORECASE)
        if MPATTERN:
            if pd.notna(num):
                current_id = str(idR) +"_"+ str(num)
            elif i in art_num:
                current_id = str(idR) +"_"+ str(art_num[i])
            else:
                current_id = str(idR) +"_"+ str(i)
                # print(current_id)
            # print(MPATTERN.group(0), file, end="\n\n")
            contents.append(file)
            tokens_nb.append(tokens)
            currents_id.append(current_id)
            keywords.append(MPATTERN.group(0))
            reference_scores.append(article_matches(file))

#save data

from pathlib import Path

output_dir = Path("legimaintlex_sample2")
output_dir.mkdir(exist_ok=True)

# Prioritize records containing the manually selected legal articles, then
# complete the sample randomly if fewer than 50 records match.
# sample_size = min(50, len(contents))
# prioritized = [i for i, score in enumerate(reference_scores) if score > 0]
# other_indices = [i for i, score in enumerate(reference_scores) if score == 0]

# if len(prioritized) > sample_size:
#     sample_indices = random.sample(prioritized, sample_size)
# else:
#     sample_indices = prioritized + random.sample(
#         other_indices, min(sample_size - len(prioritized), len(other_indices))
#     )

# sample_indices = list(contents.index())  # Use all available records for the sample

for index in range(len(contents)):
    current_id = currents_id[index]
    content = contents[index]
    file_path = output_dir / f"{current_id}.txt"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"{len(contents)} fichiers créés dans {output_dir}")
