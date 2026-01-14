from src.common.config import load_config

if __name__ == "__main__":
    cfg = load_config("src/config/default.yaml")
    print(cfg)
