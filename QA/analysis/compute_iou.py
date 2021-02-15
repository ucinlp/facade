import argparse 
import pickle 

from nltk.corpus import stopwords

from adversarial_grads.util.misc import extract_question

def compute_iou(examples_1: [str], examples_2: [str], attack_target: str):
    """
    Compute Intersection Over Union for two sets of attacked examples. 
    """
    total_iou = 0

    for ex_1, ex_2 in zip(examples_1, examples_2):
        
        if attack_target == 'question':
            tokens_1 = extract_question(ex_1)
            tokens_2 = extract_question(ex_2) 

        set_1 = set(tokens_1)
        set_2 = set(tokens_2)

        intersection = set_1.intersection(set_2)
        union = set_1.union(set_2)

        total_iou += len(intersection)/len(union)

    return total_iou/len(examples_1)

def at_least_one_stop_token(examples: [str], attack_target: str) -> int:
    """
    Return the percentage of attacked examples that contain at least one stop token.
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    for ex in examples:
        if attack_target == "question":
            ex = extract_question(ex)

        if len(set(ex).intersection(stop_tokens)) != 0:
            total += 1
        
    return total/len(examples)

def all_stop_token(examples: [str], attack_target: str) -> int:
    """
    Return the percentage of attacked examples that only contain stop tokens.
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    for ex in examples: 
        if attack_target == "question":
            ex = extract_question(ex)

        if len(set(ex).intersection(stop_tokens)) == len(set(ex)):
            total += 1

    return total/len(examples)

def stop_token_attribution(examples: [str], attack_target: str) -> int:
    """
    Return the percentage of attribution that is allocated to stop words.
    """
    stop_tokens = set(stopwords.words('english'))

    total = 0
    for ex in examples:
        if attack_target == "question":
            ex = extract_question(ex)

        hits = 0
        for token in ex: 
            if token in stop_tokens: 
                hits += 1

        total += hits/len(ex)

    return total/len(examples)

def main():
    args = argument_parsing()
    with open('./data/input_reduction_examples_{}.pkl'.format(args.id), 'rb') as f:
        data = pickle.load(f)
        baseline_ex = data[1]
        combined_ex = data[3]

    with open('attacker_metrics/input_reduction_metrics_{}'.format(args.id), 'a') as f:
        f.write("\nBaseline Stop Token Metrics\n")
        f.write("---------------------------\n")
        f.write("at_least_one_stop_token: {}\n".format(at_least_one_stop_token(baseline_ex, args.attack_target)))
        f.write("all_stop_token: {}\n".format(all_stop_token(baseline_ex, args.attack_target)))
        f.write("attribution: {}\n".format(stop_token_attribution(baseline_ex, args.attack_target)))

        f.write("\nCombined Stop Token Metrics\n")
        f.write("---------------------------\n")
        f.write("at_least_one_stop_token: {}\n".format(at_least_one_stop_token(combined_ex, args.attack_target)))
        f.write("all_stop_token: {}\n".format(all_stop_token(combined_ex, args.attack_target)))
        f.write("attribution: {}\n".format(stop_token_attribution(combined_ex, args.attack_target)))


def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--id', type=int, help='Id of files to load')
    parser.add_argument('--attack_target', type=str, choices=['question', 'paragraph'], help='Specified part for which we compute IOU')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()