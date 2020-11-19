import argparse 
import pickle 

from nltk.corpus import stopwords

def compute_iou(examples_1: [str], examples_2: [str]):
    """
    TODO 
    """
    total_iou = 0

    for ex_1, ex_2 in zip(examples_1, examples_2):
        set_1 = set(ex_1)
        set_2 = set(ex_2)

        print("set 1", set_1)
        print("set 2", set_2)

        intersection = set_1.intersection(set_2)
        union = set_1.union(set_2)

        total_iou += len(intersection)/len(union)

    return total_iou/len(examples_1)

def at_least_one_stop_token(examples: [str]) -> int:
    """
    TODO
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    for ex in examples:
        if len(set(ex).intersection(stop_tokens)) != 0:
            total += 1
        
    return total/len(examples)

def all_stop_token(examples: [str]) -> int:
    """
    TODO
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    for ex in examples: 
        if len(set(ex).intersection(stop_tokens)) == len(set(ex)):
            total += 1

    return total/len(examples)

def stop_token_attribution(examples: [str]) -> int:
    """
    TODO
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    for ex in examples:
        hits = 0
        for token in ex: 
            if token in stop_tokens: 
                hits += 1

        total += hits/len(ex)

    return total/len(examples)

def find_sick_examples(baseline, combined):
    """
    TODO 
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    idx = 0
    for base_ex, comb_ex in zip(baseline[1], combined[1]): 
        hits = 0
        for token in comb_ex: 
            if token in stop_tokens: 
                hits += 1
        # if len(set(comb_ex).intersection(stop_tokens)) > 4:
        if hits >= 2 and len(baseline[0][idx]) <= 10 and len(base_ex) >= 4:
            print("Original:", baseline[0][idx])
            print("Base reduction:", base_ex)
            print("Combined reduction:", comb_ex)
            print("Stop words:", set(comb_ex).intersection(stop_tokens))
            print()
        idx += 1

def main():
    args = argument_parsing()
    data_dir = './input_reduction_data/data_{}'.format(args.id)

    with open('{}/ir_examples_baseline_{}.pkl'.format(data_dir, args.id), 'rb') as f:
        data = pickle.load(f)
        baseline_data = data 
        baseline_ex = data[1]

    with open('{}/ir_examples_combined_{}.pkl'.format(data_dir, args.id), 'rb') as f:
        data = pickle.load(f)
        combined_ex = data[1]

    with open('{}/ir_examples_simple_combined_{}.pkl'.format(data_dir, args.id), 'rb') as f:
        data = pickle.load(f)
        simple_combined_data = data 
        simple_combined_ex = data[1]

    # with open('attacker_metrics/input_reduction_metrics_{}'.format(args.id), 'a') as f:
    #     f.write("\nBaseline Stop Token Metrics\n")
    #     f.write("---------------------------\n")
    #     f.write("at_least_one_stop_token: {}\n".format(at_least_one_stop_token(baseline_ex)))
    #     f.write("all_stop_token: {}\n".format(all_stop_token(baseline_ex)))
    #     f.write("attribution: {}\n".format(stop_token_attribution(baseline_ex)))

    #     f.write("\nCombined Stop Token Metrics\n")
    #     f.write("---------------------------\n")
    #     f.write("at_least_one_stop_token: {}\n".format(at_least_one_stop_token(combined_ex)))
    #     f.write("all_stop_token: {}\n".format(all_stop_token(combined_ex)))
    #     f.write("attribution: {}\n".format(stop_token_attribution(combined_ex)))

    #     f.write("\nSimple Combined Stop Token Metrics\n")
    #     f.write("---------------------------\n")
    #     f.write("at_least_one_stop_token: {}\n".format(at_least_one_stop_token(simple_combined_ex)))
    #     f.write("all_stop_token: {}\n".format(all_stop_token(simple_combined_ex)))
    #     f.write("attribution: {}\n".format(stop_token_attribution(simple_combined_ex)))

    find_sick_examples(baseline_data, simple_combined_data)


def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--id', type=int, help='Id of files to load')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()