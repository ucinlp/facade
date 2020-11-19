import argparse 
import pickle 

def write_ir_data(file_name, original, reduced, id):
    """
    TODO
    """
    with open('input_reduction_data/data_{}/{}'.format(id, file_name), 'wb') as f:
        pickle.dump(
            [
                original, 
                reduced, 
            ], f
        )

def main():
    args = argument_parsing()
    with open("input_reduction_data/data_{}/input_reduction_examples_{}.pkl".format(args.id, args.id), 'rb') as f:
        data = pickle.load(f)
        baseline_examples_original = data[0]
        baseline_examples_reduced = data[1]

        combined_examples_original = data[2]
        combined_examples_reduced = data[3]

    write_ir_data('ir_examples_baseline_{}.pkl'.format(args.id), baseline_examples_original, baseline_examples_reduced, args.id)
    write_ir_data('ir_examples_combined_{}.pkl'.format(args.id), combined_examples_original, combined_examples_reduced, args.id)    

    with open("input_reduction_data/data_{}/input_reduction_lengths_{}.pkl".format(args.id, args.id), 'rb') as f:
        data = pickle.load(f)
        baseline_lengths_original = data[0]
        baseline_lengths_reduced = data[1]

        combined_lengths_original = data[2]
        combined_lengths_reduced = data[3]

    write_ir_data('ir_lengths_baseline_{}.pkl'.format(args.id), baseline_lengths_original, baseline_lengths_reduced, args.id)
    write_ir_data('ir_lengths_combined_{}.pkl'.format(args.id), combined_lengths_original, combined_lengths_reduced, args.id)    
    

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--id', type=int, help='Id of the data')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()