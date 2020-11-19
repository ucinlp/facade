import argparse 
import pickle 

def write_hotflip_data(file_name, flips):
    """
    TODO
    """
    with open('hotflip_data/{}.pkl'.format(file_name), 'wb') as f:
        pickle.dump(
            [
                flips 
            ], f
        )

def main():
    args = argument_parsing()
    with open("hotflip_data/hotflip_figure_stats_{}.pkl".format(args.id), 'rb') as f:
        data = pickle.load(f)
        baseline_flips = data[0]
        combined_flips = data[1]

    write_hotflip_data("hotflip_baseline_{}".format(args.id), baseline_flips)
    write_hotflip_data("hotflip_combined_{}".format(args.id), combined_flips)

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--id', type=int, help='Id of the data')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()