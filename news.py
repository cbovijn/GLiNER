from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
text = """
Catastrophic Hurricane in Atlantica. The storm has caused widespread destruction, leaving many without homes and basic necessities. 
Power outages are widespread, and communication lines are down. There is an urgent need for food, water, and medical supplies.
"""
labels = ["disaster", "location", "situation", "supply"]
entities = model.predict_entities(text, labels)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
