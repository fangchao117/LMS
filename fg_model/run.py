"""一键：训练 + 预测。"""
from train import main as train_main
from predict import main as predict_main


if __name__ == "__main__":
    train_main()
    print()
    predict_main()
