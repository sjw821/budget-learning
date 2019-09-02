# budget-learning
## Feature budgeted Forest :
- code migrated from visual c++ to g++
- dependancies : lpsolve (included) cplex (cplex path has to be changed in makefile)
## Greedy miser:
- based on RTrank
- maximum number of features is assumed under 1000
- core written in c++ (Greedymiser/cart)
- compiled linux 64 bit binary and makefile provided
- input has to be in svmlight file format (from data directory)
- python sklearn like interface in miser.py
- results in jupyter notebook (Miser_Synth_Results)