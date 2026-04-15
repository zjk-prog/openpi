import numpy as np
import sys

def print_structure(data, indent=0):
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in data.files:
            print(" " * indent + f"├── {key}")
            print_structure(data[key], indent + 4)
    elif isinstance(data, np.ndarray):
        print(" " * indent + f"└── Shape: {data.shape}, dtype: {data.dtype}")
        if data.dtype == object and data.size == 1:
            try:
                item = data.item()
                if isinstance(item, dict):
                    for k, v in item.items():
                        print(" " * (indent + 4) + f"├── {k}")
                        print_structure(v, indent + 8)
            except Exception:
                pass
    elif isinstance(data, dict):
        for k, v in data.items():
            print(" " * indent + f"├── {k}")
            print_structure(v, indent + 4)
    else:
        print(" " * indent + f"└── Type: {type(data)}")

def main():
    # 默认路径
    file_path = "/mnt/public/zhoujiakai/0224/real_new_ctrl_pp_dice_10hz/episode_0/data.npz"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    try:
        print(f"Loading {file_path}...")
        data = np.load(file_path, allow_pickle=True)
        print("Structure:")
        print_structure(data)
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
