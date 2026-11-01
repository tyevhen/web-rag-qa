# Rolling 7-day means below these trigger a warning log (wire the log to alerting).
# Kept dependency-free so the dashboard can import it without loading ragas or torch.
ALERT_THRESHOLDS = {"faithfulness": 0.8, "answer_relevancy": 0.7}
