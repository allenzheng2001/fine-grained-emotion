# fine-grained-emotion
fine-grained emotion detection using cognitive appraisal theory

# BART/BERT:

predict multiclass based off text

# LR:
next step: read in appraisal vectors
- fillna(0)?

baseline for just appraisal vectors?

# BART/BERT revisited:

embed the appraisals into the text somehow
maybe there can be a hybrid with LR?

# LLM w/o appraisals

2-step prompting (w/ formatting?):
1. [POST], select the emotions from a set
2. from your selections {selections} of emotions, rate

# LLM w/ appraisals, CoT, self-refinement
- use appraisal embeddings? ratings, rationale, combined?
- chain of thought:
    - 7 iterations, one per emotion
    - follow up with intensity if detected
- self-refine: keep refining the answer based on appraisal

improvement, use the neuro-symbolic model?
