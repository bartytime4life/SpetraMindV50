"""
Ensemble symbolic predictor combining rule-based + NN outputs.
"""
import numpy as np

class SymbolicFusionPredictor:
    def __init__(self, rule_predictor, nn_predictor):
        self.rule_predictor = rule_predictor
        self.nn_predictor = nn_predictor

    def fuse(self, mu, nn_input):
        rule_scores = self.rule_predictor.evaluate(mu)
        nn_scores = self.nn_predictor(nn_input).detach().cpu().numpy()
        return {"rule": rule_scores, "nn": nn_scores.tolist()}
