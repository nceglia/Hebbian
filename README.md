#Hebbian
=======

__Boolean Hebbian learning framework__

###Scripts:
**_HebbianNetwork_**
####Arguments:

1. **rate**: Learning Rate
2. **sigmoid**: Defines how to sigmoid weights, options are 0, 1, 2, 3: see code.
3. **examples**: Number of random boolean examples to present
4. **hidden**: Hidden Units, 0 removes hidden layer
5. **variables**: N variable boolean functions
6. **layers**: 1 .. N layers
7. **rule**: Learning rule can either be "oja" for Oja's rule or "hebbian" for basic Hebb rule.
8. **dropout**: Edge drop out rate

####Example:

<b>
python HebbianNetwork.py  --rate 0.01 --sigmoid 1 --examples 1000 --hidden 3 --variables 2 --layer 1 --rule oja --dropout 0.5
</b>

###Tests
**_HebbianTest_**
Runs some basic test with unittest and mock library.

###Models:
  Stores current model by function.  "hebb0.txt" is the first boolean function for n variable input.
  __mkdir models__ in Hebbian directory.

###Data:
  Boolean Class creates a list of random examples for N variable boolean functions.
  
  
  
