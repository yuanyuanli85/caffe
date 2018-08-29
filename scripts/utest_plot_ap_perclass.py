import matplotlib.pyplot as plt
import numpy as np

CLASSES = ('Toy', 'Bicycle', 'Home application',
           'Couch', 'Human body', 'Plumbing fixture',
           'Bed', 'Table', 'Telephone',
           'Auto part', 'Kitchen application', 'Car')


def main(mfile):
    with open(mfile) as f:
        lines = f.readlines()

    mlist = list()
    for line in lines:
        if 'class' not in line:
            continue
        ap = line.strip().split(':')[-1]
        mlist.append(float(ap))

    ypos = np.arange(len(CLASSES))
    plt.bar(ypos, mlist)
    plt.xticks(ypos, CLASSES)
    plt.show()

if __name__ == "__main__":
    main("../../train_opim/per_class_iter_196k.log")