## Queries to view in the new ontology

ontology_properties_counts = """
PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX semleg:  <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>

SELECT (COUNT(DISTINCT ?predicate_local_name) AS ?n_distinct_predicates)
WHERE {
  ?statement a semleg:ExtractedRelation ;
             rdf:predicate ?predicate .

  FILTER(
    STRSTARTS(STR(?predicate), STR(semleg:)) ||
    STRSTARTS(STR(?predicate), STR(semlegm:))
  )

  BIND(REPLACE(STR(?predicate), "^.*/|^.*#", "") AS ?predicate_local_name)
}
"""

ontology_properites_list = """
PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX semleg:  <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>

SELECT ?predicate_local_name
       (COUNT(DISTINCT ?statement) AS ?n_examples)
WHERE {
  ?statement a semleg:ExtractedRelation ;
             rdf:predicate ?predicate .

  FILTER(
    STRSTARTS(STR(?predicate), STR(semleg:)) ||
    STRSTARTS(STR(?predicate), STR(semlegm:))
  )

  BIND(REPLACE(STR(?predicate), "^.*/|^.*#", "") AS ?predicate_local_name)
}
GROUP BY ?predicate_local_name
ORDER BY DESC(?n_examples) ?predicate_local_name
"""

## Queries to view in the KG

class_types_counts = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?class (SAMPLE(?label) AS ?classLabel) (COUNT(DISTINCT ?instance) AS ?count)
WHERE {
  ?instance rdf:type ?class .
  OPTIONAL { ?class rdfs:label ?label . }
}
GROUP BY ?class
ORDER BY DESC(?count) ?class
"""

sign = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>

SELECT ?subject_class ?predicate ?object_class
       (COUNT(DISTINCT ?statement) AS ?n_examples)
WHERE {
  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?subject ;
             rdf:predicate ?predicate ;
             rdf:object ?object .

  ?subject a ?subject_class .
  ?object a ?object_class .

  FILTER(STRSTARTS(STR(?subject_class), STR(semleg:)) || STRSTARTS(STR(?subject_class), STR(semlegm:)))
  FILTER(STRSTARTS(STR(?object_class), STR(semleg:)) || STRSTARTS(STR(?object_class), STR(semlegm:)))
}
GROUP BY ?subject_class ?predicate ?object_class
ORDER BY DESC(?n_examples) ?subject_class ?predicate ?object_class

"""

topic = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX semleg: <https://w3id.org/semleg#>

SELECT ?topic (COUNT(DISTINCT ?statement) AS ?count)
WHERE {
  ?statement a rdf:Statement ;
             a semleg:ExtractedRelation ;
             dcterms:subject ?topic .
}
GROUP BY ?topic
ORDER BY DESC(?count)
"""

maintenance_by_topic_nb = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX semleg: <https://w3id.org/semleg#>

SELECT (COUNT(DISTINCT ?article) AS ?n_articles_with_maintenance_triplets)
WHERE {
  ?statement a semleg:ExtractedRelation ;
             dcterms:subject ?topic ;
             prov:wasDerivedFrom ?article .

  FILTER(LCASE(STR(?topic)) = "maintenanceactivity")
}
"""

maintenance_by_topic_per_article = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX semleg: <https://w3id.org/semleg#>

SELECT ?article (COUNT(DISTINCT ?statement) AS ?n_maintenance_triplets)
WHERE {
  ?statement a semleg:ExtractedRelation ;
             dcterms:subject ?topic ;
             prov:wasDerivedFrom ?article .

  FILTER(LCASE(STR(?topic)) = "maintenanceactivity")
}
GROUP BY ?article
ORDER BY DESC(?n_maintenance_triplets)

"""

at_least_one_per_article = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX semleg: <https://w3id.org/semleg#>

SELECT ?topic (COUNT(DISTINCT ?article) AS ?n_articles)
WHERE {
  ?statement a semleg:ExtractedRelation ;
             dcterms:subject ?topic ;
             prov:wasDerivedFrom ?article .
}
GROUP BY ?topic
ORDER BY DESC(?n_articles)
"""

triples_by_article_topic = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX semleg: <https://w3id.org/semleg#>

SELECT ?article ?article_subject
       (COUNT(DISTINCT ?maintenance_statement) AS ?n_maintenance_triplets)
WHERE {
  {
    SELECT DISTINCT ?article ?article_subject
    WHERE {
      ?article dcterms:subject ?article_subject .
    }
    # LIMIT 100
  }

  OPTIONAL {
    ?maintenance_statement a semleg:ExtractedRelation ;
                           prov:wasDerivedFrom ?article ;
                           dcterms:subject ?triplet_topic .

    FILTER(LCASE(STR(?triplet_topic)) = "maintenanceactivity")
  }
}
GROUP BY ?article ?article_subject
ORDER BY DESC(?n_maintenance_triplets)

"""

## CQs

# CQ1: Which Roles Do Actors Play In Actions?

CQ_query = """
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?actor ?actor_label ?role_property ?action ?action_label
WHERE {
  ?actor a semleg:Actor ;
         ?role_property ?action .

  ?action a semleg:Action .

  OPTIONAL { ?actor skos:prefLabel ?actor_label . }
  OPTIONAL { ?action skos:prefLabel ?action_label . }
}
ORDER BY ?actor_label ?role_property ?action_label
"""

CQ1_query_v2 = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX semleg: <https://w3id.org/semleg#>

SELECT DISTINCT ?class_subject ?role_property ?class_object
WHERE {
  {
    ?statement a semleg:ExtractedRelation ;
               rdf:subject ?subject ;
               rdf:predicate ?role_property ;
               rdf:object ?object .

    ?subject a semleg:Actor .
    ?object a semleg:Action .

    BIND(semleg:Actor AS ?class_subject)
    BIND(semleg:Action AS ?class_object)
  }
  UNION
  {
    ?statement a semleg:ExtractedRelation ;
               rdf:subject ?subject ;
               rdf:predicate ?role_property ;
               rdf:object ?object .

    ?subject a semleg:Action .
    ?object a semleg:Actor .

    BIND(semleg:Action AS ?class_subject)
    BIND(semleg:Actor AS ?class_object)
  }
}
ORDER BY ?class_subject ?role_property ?class_object
"""

# CQ2: Under Which Conditions, Reasons, Modalities, Temporal Constraints, And Locations Must An Action Or Artifact Comply?

CQ2_query = """
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?element ?element_label ?element_type
       ?constraint_property ?constraint ?constraint_label ?constraint_type
WHERE {
  VALUES ?element_type {
    semleg:Action
    semleg:Artifact
  }

  VALUES ?constraint_type {
    semleg:Condition
    semleg:Reason
    semleg:Time
    semleg:Location
    semlegm:Modality
  }

  ?element a ?element_type ;
           ?constraint_property ?constraint .

  ?constraint a ?constraint_type .

  OPTIONAL { ?element skos:prefLabel ?element_label . }
  OPTIONAL { ?constraint skos:prefLabel ?constraint_label . }
}
ORDER BY ?element_type ?element_label ?constraint_type
"""

# CQ3: Which Legal Sources Define Or Justify The Meaning Of An Action?

CQ3_query = """
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?action ?action_label ?source_property ?source ?source_label
WHERE {
  VALUES ?source_property {
    semlegm:hasSource
    semlegm:justifiedBy
    semlegm:associatedWithModality
  }

  ?action a semleg:Action ;
          ?source_property ?source .

  ?source a semleg:Source .

  OPTIONAL { ?action skos:prefLabel ?action_label . }
  OPTIONAL { ?source skos:prefLabel ?source_label . }
}
ORDER BY ?action_label ?source_property ?source_label
"""
# CQ4: Which Elements Are Justified By Which Legal Sources?

CQ4_query = """
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?element ?element_label ?element_type
       ?source_property ?source ?source_label
WHERE {
  VALUES ?element_type {
    semleg:Action
    semleg:Condition
  }

  VALUES ?source_property {
    semlegm:hasSource
    semlegm:justifiedBy
  }

  ?element a ?element_type ;
           ?source_property ?source .

  ?source a semleg:Source .

  OPTIONAL { ?element skos:prefLabel ?element_label . }
  OPTIONAL { ?source skos:prefLabel ?source_label . }
}
ORDER BY ?element_type ?element_label ?source_property
"""

# CQ5: Which Artifacts Are Affected By Actions, And What Roles Do They Play?

CQ5_query = """
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?action ?action_label
       ?artifact_role_property
       ?artifact ?artifact_label
WHERE {
  ?action a semleg:Action ;
          ?artifact_role_property ?artifact .

  ?artifact a semleg:Artifact .

  OPTIONAL { ?action skos:prefLabel ?action_label . }
  OPTIONAL { ?artifact skos:prefLabel ?artifact_label . }
}
ORDER BY ?action_label ?artifact_role_property ?artifact_label
"""


## CQs with source article and article content

# CQ1: Which roles do actors play in actions?

CQ1_with_article_content_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX cnt: <http://www.w3.org/2011/content#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?article ?article_id_local ?article_number ?article_content
       ?direction ?actor ?actor_label ?role_property ?action ?action_label
       ?statement
WHERE {
  {
    ?statement a semleg:ExtractedRelation ;
               rdf:subject ?actor ;
               rdf:predicate ?role_property ;
               rdf:object ?action ;
               prov:hadPrimarySource ?article .

    ?actor a semleg:Actor .
    ?action a semleg:Action .
    BIND("Actor -> Action" AS ?direction)
  }
  UNION
  {
    ?statement a semleg:ExtractedRelation ;
               rdf:subject ?action ;
               rdf:predicate ?role_property ;
               rdf:object ?actor ;
               prov:hadPrimarySource ?article .

    ?action a semleg:Action .
    ?actor a semleg:Actor .
    BIND("Action -> Actor" AS ?direction)
  }

  OPTIONAL { ?actor skos:prefLabel ?actor_label . }
  OPTIONAL { ?action skos:prefLabel ?action_label . }
  OPTIONAL { ?article eli:id_local ?article_id_local . }
  OPTIONAL { ?article eli:number ?article_number . }
  OPTIONAL {
    ?article semleg:hasContent ?content_node .
    ?content_node cnt:chars ?article_content .
  }
}
ORDER BY ?article_id_local ?article_number ?direction ?actor_label ?role_property ?action_label
"""


# CQ2: Under which conditions, reasons, modalities, temporal constraints,
# and locations must a given action be performed or an artifact comply?

CQ2_with_article_content_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX cnt: <http://www.w3.org/2011/content#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?article ?article_id_local ?article_number ?article_content
       ?element ?element_label ?element_type
       ?constraint_property ?constraint ?constraint_label ?constraint_type
       ?statement
WHERE {
  VALUES ?element_type {
    semleg:Action
    semleg:Artifact
  }

  VALUES ?constraint_type {
    semleg:Condition
    semleg:Reason
    semleg:Time
    semleg:Location
    semlegm:Modality
  }

  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?element ;
             rdf:predicate ?constraint_property ;
             rdf:object ?constraint ;
             prov:hadPrimarySource ?article .

  ?element a ?element_type .
  ?constraint a ?constraint_type .

  OPTIONAL { ?element skos:prefLabel ?element_label . }
  OPTIONAL { ?constraint skos:prefLabel ?constraint_label . }
  OPTIONAL { ?article eli:id_local ?article_id_local . }
  OPTIONAL { ?article eli:number ?article_number . }
  OPTIONAL {
    ?article semleg:hasContent ?content_node .
    ?content_node cnt:chars ?article_content .
  }
}
ORDER BY ?article_id_local ?article_number ?element_type ?element_label ?constraint_type
"""


# CQ3: Which legal sources define or justify the meaning of an action?

CQ3_with_article_content_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX cnt: <http://www.w3.org/2011/content#>
PREFIX oa: <http://www.w3.org/ns/oa#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?article 
	   ?action
       ?action_label ?action_start ?action_end
       ?source_property
       ?source_label ?source_start ?source_end
       ?statement
WHERE {
  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?action ;
             rdf:predicate ?source_property ;
             rdf:object ?source ;
             prov:hadPrimarySource|prov:wasDerivedFrom ?article .

  ?action a semleg:Action .
  ?source a semleg:Source .

  OPTIONAL { ?action skos:prefLabel ?action_label . }
  OPTIONAL { ?source skos:prefLabel ?source_label . }
  OPTIONAL { ?article eli:id_local ?article_id_local . }
  OPTIONAL { ?article eli:number ?article_number . }

  OPTIONAL {
    ?article semleg:hasContent ?content_node .
    ?content_node cnt:chars ?article_content .
  }

  BIND(IRI(CONCAT(STR(?statement), "/mention/head")) AS ?action_annotation)
  OPTIONAL {
    ?action_annotation oa:hasBody ?action ;
                       oa:hasTarget ?action_target .

    ?action_target oa:hasSource ?article ;
                   oa:hasSelector ?action_selector .

    ?action_selector oa:start ?action_start ;
                     oa:end ?action_end .

    BIND(
      SUBSTR(
        STR(?article_content),
        xsd:integer(?action_start) + 1,
        xsd:integer(?action_end) - xsd:integer(?action_start)
      ) AS ?action_mention
    )
  }

  BIND(IRI(CONCAT(STR(?statement), "/mention/tail")) AS ?source_annotation)
  OPTIONAL {
    ?source_annotation oa:hasBody ?source ;
                       oa:hasTarget ?source_target .

    ?source_target oa:hasSource ?article ;
                   oa:hasSelector ?source_selector .

    ?source_selector oa:start ?source_start ;
                     oa:end ?source_end .

    BIND(
      SUBSTR(
        STR(?article_content),
        xsd:integer(?source_start) + 1,
        xsd:integer(?source_end) - xsd:integer(?source_start)
      ) AS ?source_mention
    )
  }
}
ORDER BY ?action_label ?source_property ?source_label
"""


# CQ4: Which elements (conditions, actions) are justified by which legal sources?

CQ4_with_article_content_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX cnt: <http://www.w3.org/2011/content#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?article ?article_id_local ?article_number ?article_content
       ?element ?element_label ?element_type
       ?source_property ?source ?source_label
       ?statement
WHERE {
  VALUES ?element_type {
    semleg:Action
    semleg:Condition
  }

  VALUES ?source_property {
    semlegm:hasSource
    semlegm:justifiedBy
  }

  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?element ;
             rdf:predicate ?source_property ;
             rdf:object ?source ;
             prov:hadPrimarySource ?article .

  ?element a ?element_type .
  ?source a semleg:Source .

  OPTIONAL { ?element skos:prefLabel ?element_label . }
  OPTIONAL { ?source skos:prefLabel ?source_label . }
  OPTIONAL { ?article eli:id_local ?article_id_local . }
  OPTIONAL { ?article eli:number ?article_number . }
  OPTIONAL {
    ?article semleg:hasContent ?content_node .
    ?content_node cnt:chars ?article_content .
  }
}
ORDER BY ?article_id_local ?article_number ?element_type ?element_label ?source_property
"""


# CQ5: Which artifacts are affected by actions, and what roles do they play?

CQ5_with_article_content_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX cnt: <http://www.w3.org/2011/content#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?article ?article_id_local ?article_number ?article_content
       ?action ?action_label
       ?artifact_role_property
       ?artifact ?artifact_label
       ?statement
WHERE {
  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?action ;
             rdf:predicate ?artifact_role_property ;
             rdf:object ?artifact ;
             prov:hadPrimarySource ?article .

  ?action a semleg:Action .
  ?artifact a semleg:Artifact .

  OPTIONAL { ?action skos:prefLabel ?action_label . }
  OPTIONAL { ?artifact skos:prefLabel ?artifact_label . }
  OPTIONAL { ?article eli:id_local ?article_id_local . }
  OPTIONAL { ?article eli:number ?article_number . }
  OPTIONAL {
    ?article semleg:hasContent ?content_node .
    ?content_node cnt:chars ?article_content .
  }
}
ORDER BY ?article_id_local ?article_number ?action_label ?artifact_role_property ?artifact_label
"""

## examples

# entities of a given article

query_entities_of_article = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>

CONSTRUCT {
  ?entity a prov:Entity ;
          a skos:Concept ;
          a ?semantic_role ;
          skos:prefLabel ?pref_label ;
          skos:altLabel ?alt_label .

  ?entity skos:definition ?entity_definition .
  ?entity rdfs:comment ?entity_comment .

  ?semantic_role rdfs:label ?role_label .
  ?semantic_role skos:definition ?role_definition .
  ?semantic_role rdfs:comment ?role_comment .
}
WHERE {
  VALUES ?article {
    <https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-20-novembre-2017-relatif-au-suivi-en-service-des-equipements-sous-pres-939989eef9a4/article/article-JORFTEXT000036128632-2174>
  }

  VALUES ?semantic_role {
    semleg:Actor
    semleg:Action
    semleg:Artifact
    semleg:Condition
    semleg:Reason
    semleg:Time
    semleg:Location
    semleg:Source
    semlegm:Modality
  }

  ?statement a semleg:ExtractedRelation ;
             prov:hadPrimarySource|prov:wasDerivedFrom ?article .

  {
    ?statement rdf:subject ?entity .
  }
  UNION
  {
    ?statement rdf:object ?entity .
  }

  ?entity a ?semantic_role .

  OPTIONAL { ?entity skos:prefLabel ?pref_label . }
  OPTIONAL { ?entity skos:altLabel ?alt_label . }
  OPTIONAL { ?entity skos:definition ?entity_definition . }
  OPTIONAL { ?entity rdfs:comment ?entity_comment . }

  OPTIONAL { ?semantic_role rdfs:label ?role_label . }
  OPTIONAL { ?semantic_role skos:definition ?role_definition . }
  OPTIONAL { ?semantic_role rdfs:comment ?role_comment . }
}
"""

triples_par_article = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX semleg: <https://w3id.org/semleg#>

CONSTRUCT {
  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?subject ;
             rdf:predicate ?predicate ;
             rdf:object ?object ;
             prov:hadPrimarySource ?article .

  ?subject ?predicate ?object .
}
WHERE {
  VALUES ?article {
    <https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-20-novembre-2017-relatif-au-suivi-en-service-des-equipements-sous-pres-939989eef9a4/article/article-JORFTEXT000036128632-2174>
  }

  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?subject ;
             rdf:predicate ?predicate ;
             rdf:object ?object ;
             prov:hadPrimarySource|prov:wasDerivedFrom ?article .
}
"""

query_analysis_of_signatures = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX semleg: <https://w3id.org/semleg#>
PREFIX semlegm: <https://w3id.org/semleg/maintenance#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?subject_label ?object_label
WHERE {
  ?statement a semleg:ExtractedRelation ;
             rdf:subject ?subject ;
             rdf:predicate semlegm:includesArtifact ;
             rdf:object ?object .

  ?subject a semleg:Artifact .
  ?object a semleg:Time .

  OPTIONAL { ?subject skos:prefLabel ?subject_label . }
  OPTIONAL { ?object skos:prefLabel ?object_label . }
}
ORDER BY ?subject_label ?object_label
"""
