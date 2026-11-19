import pandas as pd
import re
from collections import Counter
import random

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
            art_neww = re.sub(r"[.\s-]+", "", art_new.group(0))
            art_num[i] = art_neww
            # pass
        else:
            # print(cnt[:25])
            pass
        # current_id = str(idR) + str(num)

## data filtered

DEFAULT_PATTERN_spe = (
    r"\bcaractéristiques?\s+(?:dimensionnelles?|techniques?|"
    r"supplémentaires?|communes?|fonctionnelles?|\w+tionnel\w*|"
    r"architecturales?|constructives?|spécifiques?)\b|"
    r"\blargeurs?\b|\bdimensions?\b|\bcaractéristiques?\b|"
    r"\bhauteurs?\b|\bdistances?\b|\bsurfaces?\b|"
    r"\bminimales?\b|\bmaximales?\b|\bintérieures?\b"
)

DEFAULT_PATTERN = r"\bportes?\b|\bportails?\b|\bascenseurs?\b|\bescaliers?\b"
SPECIAL_RE = re.compile(DEFAULT_PATTERN_spe, re.IGNORECASE)
GENERAL_RE = re.compile(DEFAULT_PATTERN, re.IGNORECASE)
ASCENSEUR_RE = re.compile(r"\bascenseurs?\b", re.IGNORECASE)
contents = []
tokens_nb = []
currents_id = []
keywords = []
keywords_2 = []
nb = []
for i, row in df.iterrows():
    file = row["content"]
    topic = row["domain"]
    tokens = row["content_token_count"]
    num = row["num"]
    idR = row["id"]

    if "porte" in topic.lower() or "accessibilité" in topic.lower():
        if GENERAL_RE.search(file):
            matches = list(SPECIAL_RE.finditer(file))
            if matches:
                if pd.notna(num):
                    current_id = str(idR) + str(num)
                else:
                    current_id = str(idR) + str(i)
                    # print(current_id)
                # print(MPATTERN.group(0), file, end="\n\n")
                contents.append(file)
                tokens_nb.append(tokens)
                currents_id.append(current_id)
                keywords.append(GENERAL_RE.search(file).group(0))
                keywords_2.append(
                    ", ".join(sorted({m.group(0).lower() for m in matches}))
                )

                nb.append(
                    (
                        int(bool(ASCENSEUR_RE.search(file))),
                        len({m.group(0).lower() for m in matches}),
                        len(matches),
                    )
                )

# save data

from pathlib import Path

output_dir = Path("legimaintlex_sample_test_2")
output_dir.mkdir(exist_ok=True)

ranked_indices = sorted(range(len(contents)), key=lambda index: nb[index], reverse=True)
sample_indices = ranked_indices[:50]


def print_distribution(title, indices):
    keyword_distribution = Counter(nb[index][1] for index in indices)
    ascenseur_count = sum(nb[index][0] for index in indices)

    print(f"\n--- {title} ---")
    print(f"Articles: {len(indices)}")
    print(f"Articles avec ascenseur(s): {ascenseur_count}")
    print("Distribution per nb of keywords:")
    for number, count in sorted(keyword_distribution.items()):
        print(f"  {number} keyword(s): {count} article(s)")


print_distribution("CANDIDATS", range(len(contents)))
print_distribution("SELECTED", sample_indices)

for index in sample_indices:
    current_id = currents_id[index]
    content = contents[index]
    file_path = output_dir / f"{current_id}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"{len(sample_indices)} fichiers créés dans {output_dir}")
print(f"{len(contents)} articles candidats classés par couverture lexicale")
