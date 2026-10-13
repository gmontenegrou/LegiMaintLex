# Data Modelling
## Namespaces

```
@prefix cnt: <http://www.w3.org/2011/content#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix eli: <http://data.europa.eu/eli/ontology#> .
@prefix inst: <https://w3id.org/semleg/legimaintlex/resource/> .
@prefix oa: <http://www.w3.org/ns/oa#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <https://schema.org/> .
@prefix semleg: <https://w3id.org/semleg#> .
@prefix semlegm: <https://w3id.org/semleg/maintenance#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
```

## Source Document: LegalResource

Legal documents are represented as instances of eli:LegalResource.

* Document title (dcterms:title): full title of the legal document.

* Document type (dcterms:type): type of legal document, e.g., code, law, decree, or order.

Example:

```
<https://w3id.org/semleg/legimaintlex/resource/document/code-du-travail-857511c7b652>
    a eli:LegalResource ;
    dcterms:title "Code du travail "^^xsd:string ;
    dcterms:type "CODE"^^xsd:string .
```

## Source Article: LegalResourceSubdivision

Legal articles are represented as subdivisions of their corresponding legal resources using eli:LegalResourceSubdivision.

* Local source identifier (eli:id_local): local identifier inherited from the Légifrance source, e.g., JORFTEXT000000721798.
* Parent legal document (eli:is_part_of): URI of the legal document to which the article belongs.
* Regulatory topic (dcterms:subject): thematic category associated with the article in the corpus, e.g., Eau et santé publique.
* Source start date (oa:sourceDateStart): start date associated with the legal source.
* Article content (semleg:hasContent): URI of the resource representing the textual content of the article.

```
<https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-12-aout-1991-portant-application-de-la-directive-n-90-396-c-e-e-relati-9dc98282f046/article/article-JORFTEXT000000721798-1982>
    a eli:LegalResourceSubdivision ;
    eli:id_local "JORFTEXT000000721798"^^xsd:string ;
    eli:is_part_of <https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-12-aout-1991-portant-application-de-la-directive-n-90-396-c-e-e-relati-9dc98282f046> ;
    dcterms:subject "['Eau et santé publique']"^^xsd:string ;
    oa:sourceDateStart "1991-08-12"^^xsd:date ;
    semleg:hasContent <https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-12-aout-1991-portant-application-de-la-directive-n-90-396-c-e-e-relati-9dc98282f046/article/article-JORFTEXT000000721798-1982/content> .
```

## Article Content

The textual content of each article is represented as a cnt:ContentAsText resource.

* Textual content (cnt:chars): string containing the full textual content of the article. This separate representation facilitates the identification and positioning of entity mentions within the source text.
```
<https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-1-mars-2004-fixant-les-conditions-du-versement-de-l-acompte-de-la-cont-bda64053f117/article/article-JORFTEXT000000800025-785/content>
    a cnt:ContentAsText ;
    cnt:chars "Le ministre de la santé, de la famille et des personnes handicapées, Vu l'article 12 de la loi n° 2003-1199 du 18 décembre 2003 de financement de la sécurité sociale pour 2004 ; Vu l'avis du conseil d'administration de la Caisse nationale de l'assurance maladie des travailleurs salariés en date du 24 février 2004 ; Vu l'avis du conseil d'administration de l'Agence centrale des organismes de sécurité sociale en date du 27 février 2004, Arrête :"^^xsd:string .
```
## Conceptual Entity

Entities represent concepts, actors, artifacts, actions, conditions, or other domain elements identified during the analysis of legal texts. They are represented as skos:Concept resources, typed according to the SEMLEG ontology and modeled as prov:Entity resources for provenance purposes.

* Unique entity identifier: stable URI within the resource/entity/ namespace, generated according to the entity type and its lexical representation.
* Entity type (semleg:Condition, semleg:Actor, etc.): semantic class assigned to the extracted entity according to the ontology. In the example below, the entity is identified as a semleg:Condition.
* SKOS concept representation (skos:Concept): indicates that the entity is represented as a concept within the knowledge graph.
* Provenance entity (prov:Entity): identifies the entity as a PROV-O entity, allowing provenance information to be associated with it.
* Preferred label (skos:prefLabel): canonical label used to represent the entity in the KG.
* Alternative label (skos:altLabel): lexical variant associated with the entity. It can preserve alternative surface forms identified during extraction or entity fusion.
```
<https://w3id.org/semleg/legimaintlex/resource/entity/condition/specifications-de-la-norme-nf-en-60118-4-2007-51cd036fdf2c>
    a skos:Concept,
        prov:Entity,
        semleg:Condition ;
    skos:altLabel "spécifications de la norme NF EN 60118-4:2007"^^xsd:string ;
    skos:prefLabel "spécifications de la norme NF EN 60118-4:2007"^^xsd:string .
```
## Extracted Relations

Relations between conceptual entities are represented both as semantic relations between entities and as reified rdf:Statement resources. Reification allows each extracted relation to be represented as a first-class resource and associated with provenance information.

* Unique extracted relation identifier: stable URI within the resource/extraction/ namespace, identifying a specific extracted triple within its source article.
* Reified RDF statement (rdf:Statement): represents the extracted relation as a first-class resource so that provenance and extraction metadata can be attached to it.
* Extracted relation type (semleg:ExtractedRelation): identifies the resource as a relation produced by the information extraction process.
* Provenance entity (prov:Entity): indicates that the extracted relation is modeled as a PROV-O entity and can therefore be associated with provenance information.
* Relation subject (rdf:subject): URI of the entity acting as the subject of the extracted relation.
* Relation predicate (rdf:predicate): ontology property representing the semantic relation between the subject and object, e.g., semlegm:definedBy.
* Relation object (rdf:object): URI of the entity acting as the object of the extracted relation.
* Primary legal source (prov:hadPrimarySource): URI of the legal article from which the relation was extracted, preserving direct traceability to the supporting legal source.
* Extraction activity (prov:wasGeneratedBy): URI of the relation extraction run that generated the relation, identifying the corresponding extraction configuration or model.
```
<https://w3id.org/semleg/legimaintlex/resource/extraction/JORFTEXT000000164419/article/1er-JORFTEXT000000164419-2560/triple/0>
    a rdf:Statement,
        prov:Entity,
        semleg:ExtractedRelation ;
    rdf:object <https://w3id.org/semleg/legimaintlex/resource/entity/source/l-article-r-252-6-du-livre-ii-nouveau-du-code-rural-15069157ffa5> ;
    rdf:predicate semlegm:definedBy ;
    rdf:subject <https://w3id.org/semleg/legimaintlex/resource/entity/artifact/demandes-d-agrement-d4931ff3ab5a> ;
    prov:hadPrimarySource <https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-4-juin-1993-portant-modele-de-demande-d-agrement-des-associations-exer-7fda1335c59b/article/1er-JORFTEXT000000164419-2560> ;
    prov:wasGeneratedBy inst:relationExtractionRun_mistral_mistral_large_latest .
```

## Named Entity Annotation: Textual Mention

Annotations represent textual occurrences of conceptual entities identified within legal source documents. They follow the W3C Web Annotation Data Model and provide a precise link between a segment of the source text and its corresponding conceptual entity.

* Annotation type (oa:Annotation): represents the textual mention according to the W3C Web Annotation Data Model.
* Mentioned entity (oa:hasBody): URI of the conceptual entity (resource/entity/*) identified by the textual mention.
* Annotation target (oa:hasTarget): structure identifying the specific occurrence of the entity mention within its legal source.
* Source legal article (oa:hasSource): URI of the legal article containing the textual mention.
* Text position selector (oa:hasSelector, oa:TextPositionSelector): identifies the exact position of the mention within the source text.
* Start offset (oa:start): character offset marking the beginning of the textual mention.
* End offset (oa:end): character offset marking the end of the textual mention.
* Mention extraction activity (prov:wasGeneratedBy): URI of the extraction activity responsible for generating the annotation.
```
<https://w3id.org/semleg/legimaintlex/resource/extraction/JORFTEXT000000164419/article/1er-JORFTEXT000000164419-2560/triple/3/mention/tail>
    a oa:Annotation ;
    oa:hasBody <https://w3id.org/semleg/legimaintlex/resource/entity/actor/le-cerfa-1685f8bf601c> ;
    oa:hasTarget [
        oa:hasSelector [
            a oa:TextPositionSelector ;
            oa:end 188 ;
            oa:start 180
        ] ;
        oa:hasSource <https://w3id.org/semleg/legimaintlex/resource/document/arrete-du-4-juin-1993-portant-modele-de-demande-d-agrement-des-associations-exer-7fda1335c59b/article/1er-JORFTEXT000000164419-2560>
    ] ;
    prov:wasGeneratedBy inst:mentionExtractionActivity .
```

## Activity and KG Generator Metadata

The knowledge graph construction process is represented as a PROV-O activity. This representation records the resources used during KG generation and connects the final construction activity to the upstream extraction activities.
* Activity type (prov:Activity): identifies the knowledge graph creation process as a provenance activity according to PROV-O.
* Activity label (rdfs:label): human-readable name describing the activity, e.g., "Knowledge graph creation".
* Resources used (prov:used): references to the dataset and generator used during the KG creation process.
* Upstream extraction activities (prov:wasInformedBy): links the KG creation activity to the extraction activities that provided its inputs, including entity mention extraction and relation extraction runs.

