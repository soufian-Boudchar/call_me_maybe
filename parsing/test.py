import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--name")
parser.add_argument("--path")


args = parser.parse_args()


print(args.name)
print(args.path)
