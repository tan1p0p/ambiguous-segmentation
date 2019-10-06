import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def save_line(self, line_list, title, output_dir):
    plt.title(title)
    for line in line_list:
        plt.plot(range(len(line)), line)
    plt.savefig(os.path.join(output_dir, title + '.png'))
