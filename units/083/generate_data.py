from experiment import generate_data
import argparse, json
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--directory',default='outputs/generated-data'); a=p.parse_args()
    print(json.dumps(generate_data(a.directory),indent=2))
