# Responsible AI Design Patterns for LLM-Backed Systems

## Human-in-the-loop Enforcement
For any system that produces decisions affecting people (e.g. hiring,
moderation, approvals), the final decision field should be structurally
unable to be set automatically by the AI pipeline. One pattern is to leave
the decision field NULL in the data model until a human reviewer explicitly
calls a dedicated decision endpoint. This makes human review a architectural
guarantee rather than a policy that can be silently skipped.

## Audit Logging
All AI-assisted decisions should be logged in an append-only fashion,
capturing the model version, input, output, and any human override. Append-only
logs prevent retroactive tampering and support compliance audits.

## Bias Auditing
Before deploying a model that scores or ranks people, it is good practice to
run the same inputs through the pipeline with only demographic-signaling
attributes varied (e.g. names associated with different genders or
ethnicities) and compare output distributions. Significant divergence signals
potential bias that needs mitigation before production use.

## Guardrails Against Hallucination
Retrieval-augmented systems should explicitly detect low-confidence retrieval
(e.g. similarity scores below a threshold) and respond with an "insufficient
context" message rather than letting the LLM generate a plausible-sounding but
unsupported answer. This is especially important in domains like healthcare,
finance, or hiring where a hallucinated answer could cause real harm.

## Transparency
Systems should expose which source documents or data points contributed to a
given output, so that both users and auditors can trace a decision back to its
evidence.
