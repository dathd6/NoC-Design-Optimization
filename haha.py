import matplotlib.pyplot as plt
import numpy as np

# Sample data
np.random.seed(0)
data1 = np.random.normal(0, 1, 100)  # Data for first box plot
data2 = np.random.normal(1, 2, 100)  # Data for second box plot

# Creating the box plots
plt.boxplot([data1, data2], labels=['Data 1', 'Data 2'])
plt.title('Box Plots of Two Separate Datasets')
plt.ylabel('Values')
plt.grid(True)
plt.show()
