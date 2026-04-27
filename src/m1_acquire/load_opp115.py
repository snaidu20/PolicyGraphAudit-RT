"""
Load OPP-115 dataset from HuggingFace and print stats.
Source: alzoubi36/opp_115 on HuggingFace datasets
Paper: Wilson et al. 2016 (ACL 2016) - https://aclanthology.org/P16-1126/
"""
import json
import os

def load_categories():
    cats_path = os.path.join(os.path.dirname(__file__), "../../data/raw/opp115/categories.json")
    cats_path = os.path.abspath(cats_path)
    with open(cats_path) as f:
        return json.load(f)

def load_dataset():
    try:
        from datasets import load_dataset
        ds = load_dataset("alzoubi36/opp_115")
        return ds
    except Exception as e:
        print(f"  [WARN] datasets library error: {e}")
        return None

def main():
    print("=" * 60)
    print("OPP-115 Dataset Loader")
    print("=" * 60)

    # Load category schema
    cats = load_categories()
    print(f"\n[1] Category Schema ({cats['source']}):")
    for cat in cats['categories']:
        print(f"    [{cat['id']}] {cat['name']}")

    # Load dataset via HuggingFace
    print("\n[2] Loading HuggingFace dataset (alzoubi36/opp_115)...")
    ds = load_dataset()
    if ds is not None:
        print(f"    Splits: {list(ds.keys())}")
        for split, data in ds.items():
            print(f"    {split}: {len(data)} rows, columns={data.column_names}")
        
        # Show label distribution from train split
        if 'train' in ds:
            from collections import Counter
            labels = ds['train']['label']
            dist = Counter(labels)
            print(f"\n    Label distribution (train):")
            for label_id, count in sorted(dist.items()):
                cat_name = cats['categories'][label_id]['name'] if label_id < len(cats['categories']) else 'Unknown'
                print(f"      [{label_id}] {cat_name}: {count}")
    else:
        print("    Could not load dataset. Install: pip install datasets")

    print("\n[3] Raw data location: data/raw/opp115/categories.json")
    print("    HuggingFace: https://huggingface.co/datasets/alzoubi36/opp_115")
    print("    Total segments (all splits): ~3,432")
    print("=" * 60)

if __name__ == "__main__":
    main()
