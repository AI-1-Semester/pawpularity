## Uge 10 - Opgave 2
# Skriv en applikation, hvor man kan afprøve din model, 
# så applikationen giver mulighed for at man kan inputte data og returnerer det billede,
# der svarer til samt pawpularity score for billedet.

# Class for pictures, to generate an objekt with the picture's attributes
class PawPicture:
  def __init__(self, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur):
    self.Eyes = Eyes
    self.Face = Face
    self.Near = Near
    self.Action = Action
    self.Accessory = Accessory
    self.Group = Group
    self.Collage = Collage
    self.Human = Human
    self.Occlusion = Occlusion
    self.Info = Info
    self.Blur = Blur