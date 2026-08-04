# Similarity matrices

Both matrices are AI Hospital-specific and were generated from the 400 prepared
outputs in `embedding_inputs`.

- `diagnosis/mean_model_similarity_matrix.csv` embeds only the model's
  diagnosis conclusion, with explicit abstentions retained as full fallback
  text.
- `conversation/mean_model_similarity_matrix.csv` embeds doctor-only dialogue.

Each off-diagonal value is the mean of 100 cosine similarities, one per shared
patient. Each subdirectory also includes case-level values, raw embedded text,
normalized embedding vectors, summary statistics, and run metadata.
