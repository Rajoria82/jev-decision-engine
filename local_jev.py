import time
from typing import Dict, List, Any
from transformers import pipeline

class LocalJevEngine:
    def __init__(self, model_name: str = "valhalla/distilbart-mnli-12-3"):
        """
        Initializes the local System-1 zero-shot inference pipeline.
        This provides deterministic schema decisions without generative token overhead.
        """
        print(f"Loading local System-1 model [{model_name}]...")
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        print("Model loaded successfully.\n")

    def decide(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates input state against typed questions: Choice, Score, and Noul.
        """
        start_time = time.time()
        results = {}

        for q_id, q_def in questions.items():
            q_type = q_def.get("type")

            if q_type == "choice":
                # Multi-class categorical decision
                labels = list(q_def["criteria"].keys())
                hypothesis_template = f"{q_def.get('instructions', 'This is about')} {{}}."
                
                output = self.classifier(state, candidate_labels=labels, hypothesis_template=hypothesis_template)
                
                results[q_id] = {
                    "type": "choice",
                    "winner": output["labels"][0],
                    "confidence": round(output["scores"][0], 4),
                    "distribution": {lbl: round(score, 4) for lbl, score in zip(output["labels"], output["scores"])}
                }

            elif q_type == "noul":
                # Binary proposition probability (0.0 to 1.0)
                labels = ["true", "false"]
                hypothesis = f"Question: {q_def['instructions']} Answer: {{}}."
                
                output = self.classifier(state, candidate_labels=labels, hypothesis_template=hypothesis)
                score_dict = dict(zip(output["labels"], output["scores"]))
                prob_true = score_dict.get("true", 0.5)

                results[q_id] = {
                    "type": "noul",
                    "noul_probability": round(prob_true, 4),
                    "verdict": prob_true >= 0.50
                }

            elif q_type == "score":
                # Continuous rubric scoring mapped across scale
                levels = q_def.get("levels", ["low", "medium", "high", "critical"])
                hypothesis = f"Severity and urgency level is {{}}."
                
                output = self.classifier(state, candidate_labels=levels, hypothesis_template=hypothesis)
                
                # Weighted score calculation from 1.0 to 5.0
                weights = [i + 1 for i in range(len(levels))]
                score_dict = dict(zip(output["labels"], output["scores"]))
                expected_score = sum(score_dict[lbl] * (idx + 1) for idx, lbl in enumerate(levels))
                
                results[q_id] = {
                    "type": "score",
                    "calculated_score": round(expected_score, 2),
                    "scale": f"1 to {len(levels)}",
                    "dominant_level": output["labels"][0]
                }

        latency_ms = (time.time() - start_time) * 1000
        return {"decisions": results, "latency_ms": round(latency_ms, 2)}