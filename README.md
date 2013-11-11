Hebbian
=======

Boolean Hebbian learning framework

Scripts:
HebbianNetwork
  Arguments:
    rate: Learning Rate
    sigmoid: If "hebbian" rule is selected, defines how to sigmoid weights options are 1, 2, 3: see code.
    examples: Number of random boolean examples to present
    hidden: Hidden Units, 0 removes hidden layer
    variables: N variable boolean functions
    layers: 1 .. N layers
    algorithm: Deprecated, should be "clamp"
    update: Learning rule can either be "oja" for Oja's rule or "hebbian" for basic Hebb rule.
    
  Example:
  python HebbianNetwork.py  
          --rate 0.01 
          --sigmoid 1 
          --examples 1000 
          --hidden 3 
          --variables 2 
          --algorithm clamp 
          --layer 1 
          --update oja
Models:
  Stores current model by function.  "hebb0.txt" is the first boolean function for n variable input.

Data:
  Boolean Class creates a list of random examples for N variable boolean functions.
  
  
