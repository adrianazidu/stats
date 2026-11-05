"""
Module 6 - Customer ratings (REPEATED measures)
Question: the same 25 customers tasted 3 pizzas. Is one preferred?

Because the SAME people rate all three, the groups are not independent
(a grumpy customer gives low scores to everything). The Friedman test
handles this by ranking the pizzas within each customer.

Covers: friedmanchisquare
"""
import numpy as np
from scipy import stats
import data

margherita, quattro, diavola = data.pizza_ratings()
print("customer  margherita  4formaggi  diavola")
for i in range(5):
    print(f"{i + 1:8d}  {margherita[i]:10.0f}  {quattro[i]:9.0f}  {diavola[i]:7.0f}")
print("...")
print(f"median    {np.median(margherita):10.1f}  {np.median(quattro):9.1f}  {np.median(diavola):7.1f}")

res = stats.friedmanchisquare(margherita, quattro, diavola)
print(f"\nFriedman: statistic = {res.statistic:.2f}, p = {res.pvalue:.5f}")
print("-> at least one pizza is rated differently" if res.pvalue < 0.05
      else "-> no evidence of a preference")
