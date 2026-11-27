// Inspect graph size by node label.
MATCH (n)
RETURN labels(n) AS labels, count(*) AS node_count
ORDER BY node_count DESC;

// Inspect supported knowledge facts and their document provenance.
MATCH (entity)-[:SUPPORTED_BY]->(doc:MedicalDocument)
OPTIONAL MATCH (doc)-[:FROM_SOURCE]->(source:KnowledgeSource)
RETURN labels(entity) AS entity_labels, entity.id AS entity_id,
       entity.name AS entity_name, doc.title AS document_title,
       doc.url AS document_url, doc.page AS page,
       source.name AS source_name, source.license AS source_license
LIMIT 100;
