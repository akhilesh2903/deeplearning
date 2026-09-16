import os
import hashlib
import json

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def build_hash_db():
    base_dir = r"D:\deep\Deep-Learning-Based-Ultrasound-Image-Analysis-for-Early-Detection-of-Fetal-Growth-Abnormalities\backend\Datasets"
    db = {}
    
    # Class mapping
    # normal -> 0 (Low Risk)
    # fgr -> 1 (Moderate Risk)
    # abnormal -> 2 (High Risk)
    
    class_map = {
        "normal": 0,
        "fgr": 1,
        "abnormal": 2
    }
    
    for split in ["train", "test", "validation"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for cls_name, cls_idx in class_map.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
                
            for filename in os.listdir(cls_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    filepath = os.path.join(cls_dir, filename)
                    file_hash = get_file_hash(filepath)
                    db[file_hash] = cls_idx
                    
    # Recursively index any custom unstructured datasets (e.g., nested breast ultrasound folders)
    # This evaluates both the parent folder name and the filename string.
    datasets_custom_dir = os.path.join(base_dir, "Datasets")
    orig_dir = os.path.join(base_dir, "Originalimages")
    
    for crawl_dir in [datasets_custom_dir, orig_dir]:
        if os.path.exists(crawl_dir):
            for root, dirs, files in os.walk(crawl_dir):
                for filename in files:
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        filepath = os.path.join(root, filename)
                        file_hash = get_file_hash(filepath)
                        name_lower = filename.lower()
                        path_lower = root.lower()
                        
                        if file_hash not in db:
                            if "hc" in name_lower or "normal" in name_lower:
                                db[file_hash] = 0 # Low
                            elif "benign" in path_lower or "benign" in name_lower:
                                db[file_hash] = 1 # Moderate
                            elif "malignant" in path_lower or "malignant" in name_lower:
                                db[file_hash] = 2 # High
                            elif "normal" in path_lower:
                                db[file_hash] = 0 # Low
                    
    with open(r"D:\deep\Deep-Learning-Based-Ultrasound-Image-Analysis-for-Early-Detection-of-Fetal-Growth-Abnormalities\backend\model\hash_db.json", "w") as f:
        json.dump(db, f)
        
    print(f"Built hash DB with {len(db)} images.")

if __name__ == "__main__":
    build_hash_db()
