import argparse
import matplotlib.pyplot as plt
import os

def get_number(valfile):
    with open(valfile) as xfile:
        lines = xfile.readlines()

    scores = []
    for _line in lines:
        if 'detection_eval =' in _line:
            _line = _line.strip()
            val_score = float(_line.split('=')[-1])
            scores.append(val_score)

    return scores

def main_plot(trainfile):

    valname = os.path.basename(os.path.dirname(trainfile))
    score   = get_number(trainfile)

    iters = [ i*4000 for i in range(len(score))]
    plt.plot(iters, score, label='detection_eval')

    plt.title('Open Images Eval Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log", help='val file with validation score', required=True)

    args = parser.parse_args()
    print args

    main_plot(args.train_log)

