"""Multi-stage news pipeline.

Discovery → Triage → Publisher → Brief writers → Article writers → Assembly.
Each stage emits structured JSON; Python templates the final HTML.

See DELIVERY_PLAN.md for the full architecture.
"""
