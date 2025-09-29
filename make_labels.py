import os

def make_labels(img_dir, output_file, label_map):
    with open(output_file, "w") as f:
        for img in os.listdir(img_dir):
            if img.endswith((".jpg", ".png", ".jpeg")):
                # Example: filenames starting with "smoke_" = smoke
                if "smoke" in img.lower():
                    label = label_map["smoke"]
                else:
                    label = label_map["nosmoke"]
                f.write(f"{img} {label}\n")

if __name__ == "__main__":
    base = "C:\\Users\\RAKSHITHA\\wildfire2\\datasets"
    label_map = {"nosmoke": 0, "smoke": 1}

    make_labels(f"{base}/train", f"{base}/train_labels.txt", label_map)
    make_labels(f"{base}/valid", f"{base}/valid_labels.txt", label_map)
    make_labels(f"{base}/test", f"{base}/test_labels.txt", label_map)

    print("✅ Label files created!")
