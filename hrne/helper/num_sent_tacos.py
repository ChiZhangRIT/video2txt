''' This function finds the distribution of number of sentences in the specified description level in TACoS dataset. '''

import os
import pdb
import matplotlib.pyplot as plt
import numpy as np

desc_level = 'detailed'
data_path = '../tacos/'

num_sent = []
for root, dirs, files in os.walk(data_path + desc_level):
    for filename in files:
        if filename.endswith('.txt'):
            num_lines = sum(1 for line in open(root + '/' + filename))
            num_sent.append(num_lines)

print "In", desc_level, "description, the largest number of sentences is", sorted(num_sent)[-1]
print "Total number of", desc_level, "is", len(num_sent)
# print sorted(num_sent)

# plot histogram
plt.hist(np.asarray(num_sent), bins=len(num_sent))
plt.xlabel('num_sent');
plt.ylabel('count');
plt.title('Histogram of number of sentences');
plt.show()
