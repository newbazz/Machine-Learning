from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')
# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(k, variance, step=2, corelation=False):
	val = 1
	ys = []
	for i in range(k):
		y = random.randrange(-variance, variance) + val
		ys.append(y)

	if corelation and corelation=="pos":val+=step
	else: val-=step

	xs = [i for i in range(k)]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
	num = (mean(xs) * mean(ys)) - (mean(xs*ys))
	den = ((mean(xs)*mean(xs)) - (mean(xs*xs)))
	m = num/den
	b = mean(ys) - m*mean(xs)
	return m,b

def squared_error(ys_orig, ys_lines):
	return sum((ys_lines - ys_orig)**2)

def cofficient_of_determination(ys_orig, ys_lines):
	ys_means_line = [mean(ys_orig) for y in ys_orig]
	sqrd_err = squared_error(ys_orig, ys_lines)
	sqrd_err_y_mean = squared_error(ys_orig, ys_means_line)
	return 1 - (sqrd_err/sqrd_err_y_mean)

xs,ys = create_dataset(40, 40, 2, corelation='pos')
m,b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m*x + b) for x in xs]
r_sqrd = cofficient_of_determination(ys, regression_line)

print(r_sqrd)
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()

#you can predict x or y when other is given
#equation is y = mx + b


