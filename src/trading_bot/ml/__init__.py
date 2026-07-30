"""ML challenger — the first *learned* strategy in the bot.

A LightGBM 5-class model trained on point-in-time OHLCV features,
deployed at shadow tier so the existing daily grading and weekly
evolution machinery judges ML vs LLM head-to-head, forward, out of
sample.

Modules:
  data      — training snapshot store (state/ml/train.db) + audit + backfill
  features  — point-in-time feature pipeline shared by train and serve
  labels    — open→close labels matching meta.reflection's grader exactly
  splits    — purged + embargoed walk-forward splitter
  train     — LightGBM walk-forward trainer, baselines, model card
"""
