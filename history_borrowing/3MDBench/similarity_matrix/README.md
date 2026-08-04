# 3MDBench assessment similarity

`assessment/` contains the only similarity matrix supported by the updated
80-case source. Each text is a canonical representation of assessment status,
seven binary rubric fields, and Overall Clinical Competence. Missing fields are
explicit tokens rather than silently omitted.

The matrix is the mean of same-case cosine similarities across 80 cases using
Qwen3-Embedding-0.6B. It measures evaluator-profile similarity, not doctor
dialogue or diagnosis similarity.
