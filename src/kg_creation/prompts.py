# ---
# ONE SHOT EXAMPLES FOR PROMPTING
# ---

# Joint extraction

EXAMPLE_CONTEXT = "La vérification des dispositifs de levage est effectuée par un technicien de maintenance tous les douze mois conformément à l'article L. 512-5 du Code de l'environnement"

EXAMPLE_TRIPLES = """ 
{{
  "triples": [
    {{
      "head": "La vérification des dispositifs de levage",
      "head_type": "Action",
      "relation": "performedBy",
      "tail": "un technicien de maintenance",
      "tail_type": "Actor"
    }},
    {{
      "head": "La vérification des dispositifs de levage",
      "head_type": "Action",
      "relation": "hasTime",
      "tail": "tous les douze mois",
      "tail_type": "Time"
    }},
    {{
      "head": "La vérification des dispositifs de levage",
      "head_type": "Action",
      "relation": "hasSource",
      "tail": "l'article L. 512-5 du Code de l'environnement",
      "tail_type": "Source"
    }}
  ]
}}
"""
ENTITIES_EXAMPLES = """
{{
  "entities": [
    {{
      "entity_id": "E1",
      "name": "Vérification des dispositifs de levage",
      "type": "Action",
      "evidence": "La vérification des dispositifs de levage"
    }},
    {{
      "entity_id": "E2",
      "name": "Technicien de maintenance",
      "type": "Actor",
      "evidence": "un technicien de maintenance"
    }},
    {{
      "entity_id": "E3",
      "name": "tous les douze mois",
      "type": "Time",
      "evidence": "tous les douze mois"
    }},
    {{
      "entity_id": "E4",
      "name": "article L. 512-5 du Code de l'environnement",
      "type": "Source",
      "evidence": "l'article L. 512- du Code de l'environnement"
    }},
  ]
}}
"""

EXAMPLE_TOPIC_CLASSIFICATION = """
{{
  "topic_by_triplet_index": [
    {{ "i": 0, "topic": "maintenanceActivity" }},
    {{ "i": 1, "topic": "maintenanceActivity" }},
    {{ "i": 2, "topic": "legalCrossReference" }}
  ]
}}
"""

COMPETENCY_QUESTIONS_EXAMPLE = """
CQ1: Which roles do actors play in actions ?
CQ2: Under which conditions, reasons, modalities, temporal constraints, and locations must a given action be performed or an artifact comply?
CQ3: Which legal sources define or justify the meaning of an action?
CQ4: Which elements (conditions, actions) are justified by which legal sources? 
CQ5: Which artifacts are affected by actions, and what roles do they play?
"""

EXAMPLE_CANDIDATES_OBJECT_PROPS = """
{{
  "candidates_object_props": [
    {{
      "object_prop_candidate": "performedBy",
      "domain": "Action",
      "range": "Actor",
      "textual_evidence": [
        "L'inspection est réalisée par un technicien qualifié."
      ],
      definition: ""Relates an action to the actor that carries it out",
      "aligned_CQ": "CQ1",
      "confidence": 0.95
    }}
  ]
}}
"""

# ---
# TEMPLATES PROMPTS
# ---


def build_prompt_triplets_extraction(
    context: str, constraints: str, domain="regulatory industrial maintenance"
) -> str:
    return f"""
# Role:
You are an expert assistant in ontology engineering, specialized in knowledge extraction from regulatory texts in {domain}.

# Task:
From CONTEXT, extract legally relevant semantic triples.

# Semantic Scope Constraints:
{constraints}

# Extraction Rules:

## Class Constraints
- Identify entities and assign them a semantic class.
- "head_type" and "tail_type" MUST be selected ONLY from the classes defined in the semantic scope constraints.
- Use the semantic scope constraints descriptions to assign the most appropriate type.

## Relation Extraction (Open)
- Extract each meaningful relation between entities as one triplet.
- The relation must be inferred from the text.
- Use concise relation labels in English and in lowerCamelCase (e.g., performedBy, hasTime).
- Avoid vague or generic relation names.

## General Constraints
- Do NOT invent information not present in the text.
- If a time, location, or legal reference modifies an entity (action, situation, or condition), create a separate triplet.
- triples MUST contain "head", "relation", and "tail". Otherwise, discard them.
- Do NOT output duplicate or redundant triples.
- Output ONLY valid JSON following this schema:

{{
  "triples": [
    {{
      "head": "...",
      "head_type": "...",
      "relation": "...",
      "tail": "...",
      "tail_type": "..."
    }}
  ]
}}
- If no meaningful relation is found, return:
  {{"triples": []}}

# Example:

CONTEXT: {EXAMPLE_CONTEXT}

TRIPLES:
{EXAMPLE_TRIPLES}

# Input

CONTEXT:
{context}

TRIPLES:
"""


# ---
# NER + RE
# ---


def build_prompt_entities_extraction_class_restricted(
    context: str, classes_constraints: str, domain="regulatory industrial maintenance"
) -> str:
    return f"""
# Role:
You are an expert assistant in ontology engineering, specialized in knowledge extraction from regulatory texts in {domain}.

# Task:
From CONTEXT, extract entities using ONLY classes allowed in the semantic scope constraints.

# Semantic Scope Constraints:
{classes_constraints}

# Extraction Rules:

- Extract all explicit entities mentioned in text (no hallucinations).
- Use the descriptions in semantic scope constraints to assign exactly the most appropriate type.
- Resolve pronouns to explicit mentions when possible.
- Do not create duplicates (same mention + same class).
- If no valid entities exist, return {{"entities": []}}.

# Output (STRICT JSON only):
{{
  "entities": [
    {{
      "entity_id": "E1",
      "name": "...",
      "type": "...",
      "evidence": "short exact text span"
    }}
  ]
}}

# Example:

CONTEXT: {EXAMPLE_CONTEXT}

ENTITIES:

{ENTITIES_EXAMPLES}

# Input

CONTEXT:
{context}

ENTITIES:
"""


# not fully


def build_prompt_relations_extraction_fully_constrained_with_entities(
    context: str,
    entities_json: str,
    constraints_full: str,
    domain="regulatory industrial maintenance",
    relation_extraction_open=True,
) -> str:

    if relation_extraction_open:
        relation_extraction_rules = """

## Relation Extraction (Open)
- Extract each meaningful relation between entities as one triplet.
- The relation must be inferred from the text.
- Use concise relation labels in English and in lowerCamelCase (e.g., performedBy, hasTime).
- Relation labels must be semantically consistent with the entity types they connect.
  Example: if the relation is hasTime, the tail_type must be Time; if the relation is hasActor, the tail_type must be Actor.
  """
    else:
        relation_extraction_rules = """
## Relation Extraction (Fully Constrained)
- The relation MUST be chosen only from the allowed in the semantic scope constraints.
- Keep the original relation as they are defined in the Constraints (in English and in lowerCamelCase).
    """
    return f"""
# Role:
You are an expert assistant in ontology engineering, specialized in knowledge extraction from regulatory texts in {domain}.

# Task:
From CONTEXT and PROVIDED ENTITIES, extract all the expressed relational triples.

# Semantic Scope Constraints:
{constraints_full}

# Extraction Rules:

{relation_extraction_rules}

## General Constraints
- Do NOT invent information not present in the text.
- triples MUST contain "head", "relation", and "tail". Otherwise, discard them.
- Do NOT output duplicate, redundant or reflexive triples.
- If no meaningful triple is found, return:
- confidence score is epistemic confidence of ontology mapping, float in [0.0, 1.0].
- If none, return {{"triples": []}}.
- Output ONLY valid JSON following this schema:

{{
  "triples": [
    {{
      "head_id": "E1",
      "head": "...",
      "head_type": "...",
      "relation": "...",
      "tail_id": "E2",
      "tail": "...",
      "tail_type": "...",
      "confidence": 0.0,
      "evidence": "short exact text span"
    }}
  ]
}}

# Example:

CONTEXT: {EXAMPLE_CONTEXT}

PROVIDED ENTITIES:

{ENTITIES_EXAMPLES}

TRIPLES:
{EXAMPLE_TRIPLES}

# Input

CONTEXT:
{context}

PROVIDED ENTITIES:
{entities_json}

TRIPLES:
"""


def build_prompt_topic_classification(
    context: str, triplets_json_str: str, domain="regulatory industrial maintenance"
) -> str:
    return f"""
# Role:
You are an expert assistant in ontology engineering, specialized in knowledge extraction from regulatory texts in {domain}.

# Task:
Classify the topic of each provided triplet based on its semantic meaning.

# Topic Definitions:
- legalCrossReference: The triplet refers primarily to another legal article, decree, regulation, or provision.
- maintenanceActivity: The triplet concerns any maintenance activity, for example, inspection, repair, servicing, maintenance operations, technical interventions, equipment upkeep, or legal compliance linked to maintenance tasks.
- anotherLegalActivity: Any legal action not directly related to maintenance activities.

# Classification Rules:
- For each input triplet, assign exactly one topic.
- Use the semantic meaning of each compact triplet to classify it.
- Output ONLY valid JSON following this schema:
{{
  "topic_by_triplet_index": [
    {{ "i": 0, "topic": "..." }},
    {{ "i": 1, "topic": "..." }}
  ]
}}

- If input triples are empty, return:
  {{
    "topic_by_triplet_index": []
  }}

# Example:

TRIPLES:
{EXAMPLE_TRIPLES}

TOPIC CLASSIFICATION:

{EXAMPLE_TOPIC_CLASSIFICATION}

# Input

TRIPLES:
{triplets_json_str}

TOPIC CLASSIFICATION:
"""


def prompt_object_properties_candidates(
    current_ontology,
    triplets_json,
    element_to_extract="object properties",
    domain="regulatory industrial maintenance",
    competency_questions=COMPETENCY_QUESTIONS_EXAMPLE,
):
    return f"""

# Role:
You are an expert assistant in ontology engineering, specialized in knowledge extraction from regulatory texts in {domain}.

# Task:
Propose candidates for {element_to_extract} to EXTEND (not replace) the existing ontology, grounded in the provided triples and explicitly guided by the competency questions.

# Critical Principles (MANDATORY):
- You MUST extend the CURRENT ontology while preserving its structure and consistency.
- Reuse ONLY existing classes for domain and range.
- Each proposed {element_to_extract} MUST help answer at least one competency question.
- Prefer semantically explicit relations (e.g., performedBy, deliveredTo, ensures, requires).
- Use triples as evidence, but DO NOT blindly transform them into relations.

# Rules for candidate generation:
- The {element_to_extract} MUST be in English, lowerCamelCase, with no underscores or spaces.
- Avoid duplicates and near-duplicates.
- confidence score is epistemic confidence of ontology extension, float in [0.0, 1.0].
- definition is a concise natural language description of the candidate relation and its intended meaning.
- The definition should be suitable as an rdfs:comment.
# Output ONLY valid JSON following this schema:
{{
  "candidates_object_props": [
    {{
      "object_prop_candidate": "...",
      "domain": "...",
      "range": "...",
      "textual_evidence": ["...", "..."],
      "definition": "...",
      "aligned_CQ": "CQx",
      "confidence": 0.0
    }}
  ]
}}

# If no valid candidate exists, return exactly:
{{"candidates_object_props": []}}

# Example:

{EXAMPLE_CANDIDATES_OBJECT_PROPS}

# Input:

CURRENT ONTOLOGY:
{current_ontology}

COMPETENCY QUESTIONS:
{competency_questions}

TRIPLES:
{triplets_json}

# CANDIDATES FOR {element_to_extract}:
"""