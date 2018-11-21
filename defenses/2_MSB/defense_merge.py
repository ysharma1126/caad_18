import argparse

import numpy as np
import pandas as pd


def run(tmp_dir, output_file):
    pred1 = pd.read_csv(tmp_dir + "/pred_1.csv", header=None)
    pred1.columns = ["picture", "prob1"]
    pred2 = pd.read_csv(tmp_dir + "/pred_2.csv", header=None)
    pred2.columns = ["picture", "prob2"]
    pred3 = pd.read_csv(tmp_dir + "/pred_3.csv", header=None)
    pred3.columns = ["picture", "prob3"]
    pred4 = pd.read_csv(tmp_dir + "/pred_4.csv", header=None)
    pred4.columns = ["picture", "prob4"]
    pred5 = pd.read_csv(tmp_dir + "/pred_5.csv", header=None)
    pred5.columns = ["picture", "prob5"]


    rs = pd.merge(pred1, pred2, on=['picture'])
    rs = pd.merge(rs, pred3, on=['picture'])
    rs = pd.merge(rs, pred4, on=['picture'])
    rs = pd.merge(rs, pred5, on=['picture'])

    avoid = True
    def get_label(row):
        prob1 = row["prob1"].split(" ")
        prob2 = row["prob2"].split(" ")
        prob3 = row["prob3"].split(" ")
        prob4 = row["prob4"].split(" ")
        prob5 = row["prob5"].split(" ")

        prob1 = np.array([float(e) for e in prob1])
        prob2 = np.array([float(e) for e in prob2])
        prob3 = np.array([float(e) for e in prob3])
        prob4 = np.array([float(e) for e in prob4])
        prob5 = np.array([float(e) for e in prob5])


        prob = (prob1 + prob2 + prob3 + prob4 + prob5) / 5.0

        main_label = np.argmax(prob)
        fool_label = 612  # index 0 611: 'jigsaw puzzle'

        label = main_label
        
        if avoid and main_label == fool_label: 
            ordered_labels = prob.argsort()[-2:][::-1]
            label = ordered_labels[1]  # The second label

        return label

    rs['label'] = rs.apply(lambda x: get_label(x), axis=1)
    rs[["picture", "label"]].to_csv(output_file, index=None, header=None)


def main():
    parser = argparse.ArgumentParser(description='Merge 5 prediction files')

    # define command line arguments
    parser.add_argument('--tmp_dir', dest='tmp_dir', type=str,
                        help='The temporary folder path', required=True)
    parser.add_argument('--output_file', dest='output_file', type=str,
                        help='The output file path', required=True)

    # parse the arguments
    args = vars(parser.parse_args())
    tmp_dir = args.get('tmp_dir')
    output_file = args.get('output_file')

    # execute
    run(tmp_dir, output_file)


if __name__ == '__main__':
    main()
