import json
from .model import train

if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=True, indent=2))
