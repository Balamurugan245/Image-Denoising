import sys
from overall import run


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python run.py train")
        print("python run.py inference")

    else:
        mode = sys.argv[1]

        run(mode)
