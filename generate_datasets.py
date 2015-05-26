

def save_samples(X, y, labels, output):
	#TODO: adjust
    np.savez(output, X=X, y=y, labels=labels)

def load_estimates(fname):
    with open(fname, 'rb') as fin:
        e = pickle.load(fin)
    return e




if __name__ == '__main__':
import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='resample.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Resample stimulus classes',
            epilog="""Example usage:

$ python resample.py 1000 estimated.pkl /path/to/output

will generate 1000 samples for each training items. The total number of output samples
is nsamples * nclasses.

Generates samples per condition

The output consists of a set of .npz files, two files per condition:
one for the training phase and one for the test phase

1. trainfile with 3 arrays
X : nsamples*nclasses x ncepstra
y : nsamples*nclasses
labels : nclasses
(zoals resampler.py)

2. testfile with 4 arrays
X1 : nsamples*nclasses x ncepstra
X2 : nsamples*nclasses x ncepstra
y : nsamples*nclasses (if y = 1, X1 is correct, if y=2, x2 is correct)

labels is a 1d array of strings, indicating how the classes (phones,
conditions) map to the label-indices in y.

            """)
        parser.add_argument('nsamples', metavar='NSAMPLES',
                            nargs=1,
                            help='number of samples per class')
        parser.add_argument('estimates', metavar='ESTIMATES',
                            nargs=1,
                            help='estimated normals in .pkl format')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='name of output file')
        parser.add_argument('-s', '--shrink',
                            action='store',
                            dest='shrink',
                            default=0,
                            help='covariance shrinkage factor')
        parser.add_argument('-d', '--dispersal',
                            action='store',
                            dest='dispersal',
                            default=1,
                            help='mean dispersal factor')
        return vars(parser.parse_args())

    args = parse_args()

    nsamples = int(args['nsamples'][0])
    input_fname = args['estimates'][0]
    output = args['output'][0]

    shrink = float(args['shrink'])
    dispersal = float(args['dispersal'])

    estimates = load_estimates(input_fname)
    X, y, labels = generate_train(estimates, nsamples, dispersal, shrink)
    save_samples(X, y, labels, output)

    X1, X2, y = generate_test(estimates, nsamples, dispersal, shrink)
    save_samples(X1, X2, labels, output)

