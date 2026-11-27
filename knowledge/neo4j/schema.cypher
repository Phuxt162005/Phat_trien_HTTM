// Idempotent schema only. No medical facts are inserted by this file.
CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (n:Symptom) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT risk_factor_id IF NOT EXISTS FOR (n:RiskFactor) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT diagnostic_test_id IF NOT EXISTS FOR (n:DiagnosticTest) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT complication_id IF NOT EXISTS FOR (n:Complication) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT treatment_id IF NOT EXISTS FOR (n:Treatment) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT medication_id IF NOT EXISTS FOR (n:Medication) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT nutrition_id IF NOT EXISTS FOR (n:Nutrition) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT medical_document_id IF NOT EXISTS FOR (n:MedicalDocument) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT document_chunk_id IF NOT EXISTS FOR (n:DocumentChunk) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT knowledge_source_id IF NOT EXISTS FOR (n:KnowledgeSource) REQUIRE n.id IS UNIQUE;
CREATE FULLTEXT INDEX diabetes_knowledge_fulltext IF NOT EXISTS
FOR (n:Disease|Symptom|RiskFactor|DiagnosticTest|Complication|Treatment|Medication|Nutrition|MedicalDocument|DocumentChunk)
ON EACH [n.name, n.title, n.description, n.text];

// Default vector index for intfloat/multilingual-e5-small (384 dimensions).
CREATE VECTOR INDEX diabetes_document_chunk_vector IF NOT EXISTS
FOR (n:DocumentChunk) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};
