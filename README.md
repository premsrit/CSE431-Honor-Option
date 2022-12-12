# CSE431-Honor-Option
This repository is part of CSE431 Honor Option to analyze performance of splay tree.  
Binary search trees used inside this experiment are normal binary search tree, AVL tree, 
red-black tree, and splay tree.

## Command to run experiment

```
python main.py --trail 10 --adding_rate 0.3 --repeating_rate 0.8
```
- trial: number trail of each experiment
- adding_rate: rate of adding operation to be included
- searching_rate: rate of searching operation to be included
- repeating_rate: rate of repeat access to recently accessed values
- figure_title: title of plot figure
- figure_name: name of saved figure file