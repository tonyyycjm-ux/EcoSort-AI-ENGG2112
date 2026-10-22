"""Source-linked household guidance, separate from visual model predictions."""
VERIFIED = "2026-10-22"
RECYCLE = "https://www.cityofsydney.nsw.gov.au/waste-recycling-services/recycle-household-items"
GENERAL = "https://www.cityofsydney.nsw.gov.au/explainers/2026/moving-in-or-out-waste-and-recycling-guide"
DOORSTEP = "https://www.cityofsydney.nsw.gov.au/environmental-support-funding/doorstep-recycling"
FOOD = "https://www.cityofsydney.nsw.gov.au/waste-recycling-services/prepare-for-your-food-scraps-recycling-collection"
GARDEN = "https://www.cityofsydney.nsw.gov.au/waste-recycling-services/recycle-garden-organics"
EPA = "https://www.epa.nsw.gov.au/Your-environment/Recycling-and-reuse/household-recycling-overview/community-recycling-centres"
EPA_OVERVIEW = "https://www.epa.nsw.gov.au/Your-environment/Recycling-and-reuse/household-recycling-overview"

ITEM_TYPES = ["Not confirmed", "Empty plastic water/drink bottle", "Empty non-black rigid plastic food container", "Empty glass bottle or jar", "Empty aluminium or steel can", "Paper or cardboard with food scraps removed", "Ceramic crockery", "Soft plastic bag or wrapper", "Polystyrene packaging", "Black plastic container", "Food scraps", "Garden leaves or small prunings", "Battery", "Small electronics", "Textiles", "Unwanted medicines", "Other / unsure"]

def disposal_guidance(label, item_type="Not confirmed", food_service=False, uncertain=False):
    if item_type not in ITEM_TYPES:
        raise ValueError("Unknown item confirmation")
    confirmed = item_type != "Not confirmed"
    source = RECYCLE
    if item_type in ITEM_TYPES[1:6]:
        route, advice = "Recyclable - yellow lid bin", "Empty and dry the accepted container, or remove food scraps from paper/cardboard. Leave bottle lids and labels on."
    elif item_type in {"Soft plastic bag or wrapper", "Black plastic container"}:
        route, advice, source = "General waste - red lid bin", "Not accepted in the yellow lid bin. Use the red lid household rubbish bin.", GENERAL
    elif item_type == "Polystyrene packaging":
        route, advice, source = "Special recycling - check collection conditions", "Clean, dry, rigid packaging foam may qualify for doorstep recycling. Food trays and bendy foam do not qualify for that service.", DOORSTEP
    elif item_type == "Battery" or (not confirmed and label == "battery"):
        route, advice, source = "Problem waste - battery drop-off", "Never put batteries in household bins. Use an appropriate battery drop-off or CRC; confirm acceptance for large or damaged batteries.", EPA
    elif item_type == "Small electronics" or (not confirmed and label == "ewaste"):
        route, advice, source = "Special recycling - e-waste collection", "Not for the yellow lid bin. Check council collection size limits and delete personal device data before recycling.", DOORSTEP
    elif item_type == "Unwanted medicines" or (not confirmed and label == "medicines"):
        route, advice, source = "Special disposal - community pharmacy", "Take unwanted medicines to a community pharmacy. Confirm packaging and sharps separately; do not infer safe disposal from packaging appearance.", EPA_OVERVIEW
    elif not confirmed and label in {"household_problem_waste", "light_bulbs"}:
        route, advice, source = "Problem waste - identify item and collection point", "Do not put in household recycling. Confirm the exact item and use an accepting CRC, Chemical CleanOut or council special service.", EPA
    elif item_type == "Food scraps" or (not confirmed and label == "food_organics"):
        if food_service:
            route, advice = "Organics - food scraps service", "Use the maroon lid food scraps bin for accepted unpackaged food. No plastic packaging, garden waste or hard shells."
        else:
            route, advice = "Food organics - confirm service; otherwise red lid", "Check access to a household or community food scraps service. Without a suitable service, food goes in general waste, not the garden organics bin."
        source = FOOD
    elif item_type == "Garden leaves or small prunings" or (not confirmed and label == "vegetation"):
        route, advice, source = "Garden organics - green lid bin", "For leaves, grass and accepted small prunings. Not food, plant pots, soil, manufactured wood or large logs.", GARDEN
    elif item_type == "Textiles" or (not confirmed and label == "textile_trash"):
        route, advice, source = "Reuse or special textile recycling", "Donate reusable items. Eligible unusable textiles must be clean and dry for doorstep collection; check exclusions. Not yellow lid recycling.", DOORSTEP
    elif item_type == "Ceramic crockery" or (not confirmed and label == "ceramics"):
        route, advice = "Non-recyclable in yellow lid - confirm disposal", "Crockery and ceramics are not accepted in yellow lid recycling. Consider reuse or check council item-specific disposal; do not treat ceramic as recyclable container glass."
    elif not confirmed and label in {"plastic", "glass", "metal", "paper", "cardboard"}:
        route, advice = "Potentially recyclable - confirm item type", "Material alone is insufficient. Yellow lid accepts specified bottles, jars, cans, paper and cardboard, not every object made from these materials. Select the matching confirmed item type."
    elif not confirmed and label == "wood":
        route, advice, source = "Wood item - check council disposal", "Manufactured wooden items are not accepted in the green lid garden organics bin. Check reuse, bulky collection or item-specific disposal.", GARDEN
    else:
        route, advice, source = "Manual review - identify exact item", "Mixed or unidentified objects need item-specific checking. Do not assume general rubbish or recycling when batteries, chemicals or sharps may be present.", RECYCLE
    if uncertain and not confirmed:
        route = "Manual review - disposal not confirmed"
        advice = "The visual classification is uncertain. Confirm the item before following a disposal route. " + advice
    heading = "User-confirmed item guidance" if confirmed else "Conditional guidance for the model suggestion"
    markdown = f"### {route}\n\n{heading}. {advice}\n\n[Official source]({source}) | City of Sydney / NSW EPA | Checked {VERIFIED}\n\nGuidance is conditional on the item and its contents. The image model cannot verify emptiness, contamination, battery condition or hidden hazards."
    return route, markdown

if __name__ == "__main__":
    assert "yellow" in disposal_guidance("plastic", ITEM_TYPES[1])[0]
    assert "red" in disposal_guidance("plastic", "Soft plastic bag or wrapper")[0]
    assert "drop-off" in disposal_guidance("battery")[0]
    assert "Manual review" in disposal_guidance("plastic", uncertain=True)[0]
    assert "pharmacy" in disposal_guidance("medicines")[0]
    assert "Potentially" in disposal_guidance("plastic")[0]
    assert "green" in disposal_guidance("vegetation")[0]
    print("Disposal rule checks passed.")
