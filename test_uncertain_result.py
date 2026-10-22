"""Check display decisions without loading or changing model weights."""
from ecosort_runtime import EcoSort


engine = EcoSort.__new__(EcoSort)
engine.predict = lambda image: ("plastic", .35, True, {"plastic": .35, "trash": .65})
scores, decision, guidance = engine.classify(object())
assert decision.startswith("Unable to identify reliably. Top candidate:")
assert "disposal not confirmed" in decision
assert scores == {"Plastic": .35, "Other waste": .65}
engine.predict = lambda image: ("plastic", .8, False, {"plastic": .8, "trash": .2})
assert engine.classify(object())[1].startswith("Model suggestion:")
print("Uncertain and confident display checks passed.")
