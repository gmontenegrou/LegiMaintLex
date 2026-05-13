# Experiments

Full data is available at [Google Drive](https://drive.google.com/drive/folders/1GfJPdJiGv_dZuC9utOgCTyzmeYb5lYU0?usp=drive_link).

```text
exp/
|-- eval/
|-- kg/
|   `-- rdf_nrows_100/
|       |-- combined/
|       |-- mistral/
|       `-- openai/
`-- new_ontology/
    |-- combined/
    |-- mistral/
    `-- openai/
```

## `eval/`

Contains CSV with the mannual evaluation files for property signatures excluded during tail-class
filtering. These files record signatures extracted from the KG whose tail class
does not match the class expected for the property.

- `ontology_new_mistral_from_kg_tail_class_filter_excluded_signatures.csv`:
  excluded signatures for the Mistral-based variant.
- `ontology_new_openai_from_kg_tail_class_filter_excluded_signatures.csv`:
  excluded signatures for the OpenAI-based variant.

<!-- The columns include the property, domain, range, expected tail class, evaluation
status, manual annotation, and support counts by signature and by property. -->

## `kg/rdf_nrows_100/`

Contains RDF graphs in Turtle (`.ttl`) generated from a 100-row sample. The
outputs use vocabularies such as ELI, PROV, SKOS, Web Annotation, and the
`semleg` / `semlegm` ontologies to represent legal documents, articles,
extracted entities, relations, and provenance.

- `mistral/LegiMaintLex_mistral_fused_100.ttl`: fused RDF KG generated from the
  Mistral extraction.
- `openai/LegiMaintLex_openai_fused_100.ttl`: fused RDF KG generated from the
  OpenAI extraction.
- `combined/LegiMaintLex_combined_fused_100.ttl`: fused RDF KG for the combined
  variant.

## `new_ontology/`

Contains extended ontologies in Turtle (`.ttl`) produced by the ontology
discovery and extension process from extracted triples.

### `new_ontology/mistral/`

- `ontology_extended_mistral.ttl`: final extended SemLeg ontology for the
  Mistral variant.
- `semleg-triple-ont-auto-extended-mistral.ttl`: automatic triple-based
  extension output for the Mistral variant.

### `new_ontology/openai/`

- `ontology_extended_openai.ttl`: final extended SemLeg ontology for the OpenAI
  variant.
- `semleg-triple-ont-auto-extended-openai.ttl`: automatic triple-based extension
  output for the OpenAI variant.

### `new_ontology/combined/`

- `ontology_extended_combined.ttl`: extended ontology for the combined variant,
  built from the available provider outputs.
