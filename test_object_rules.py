"""Small safety regression checks, without loading either image model."""
from sydney_rules import predicted_item_guidance

assert "yellow" in predicted_item_guidance("plastic", "plastic drink bottle")[0]
assert "red" in predicted_item_guidance("plastic", "plastic bag")[0]
assert "Potentially" in predicted_item_guidance("plastic", "plastic household item")[0]
assert "red" in predicted_item_guidance("trash", "used tissue")[0]
assert "drop-off" in predicted_item_guidance("battery", "household battery")[0]
assert "Manual review" in predicted_item_guidance("trash", "used tissue", uncertain=True)[0]
assert "User-confirmed" not in predicted_item_guidance("plastic", "plastic drink bottle")[1]
assert "Confirm" in predicted_item_guidance("plastic", "plastic food container")[1]
assert "otherwise red" in predicted_item_guidance("food_organics", "fruit peel")[0]
assert "maroon" in predicted_item_guidance("food_organics", "fruit peel", food_service=True)[1]
print("Object-level conditional routing and uncertainty checks passed.")
