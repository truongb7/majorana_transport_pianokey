import numpy as np
import sys as sy

val_start = float(sy.argv[1])
val_end = float(sy.argv[2])
nbr_vals = int(sy.argv[3])

arr_linspace = np.linspace(val_start, val_end, nbr_vals)
arr_linspace_round = np.round(arr_linspace, 6)
print(arr_linspace_round)