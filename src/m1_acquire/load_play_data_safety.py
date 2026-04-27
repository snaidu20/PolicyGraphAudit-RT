"""
Load Google Play Store Data Safety labels dataset and print stats.
Source: WIPI/GoogleDataSafety on HuggingFace
https://huggingface.co/datasets/WIPI/GoogleDataSafety
12.97M rows, collected 2022-2023. License: CC-BY-NC-4.0.
"""
import json
import os

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/play_data_safety"))

def main():
    print("=" * 60)
    print("Google Play Data Safety Labels Loader")
    print("=" * 60)

    # Load schema
    schema_path = os.path.join(RAW_DIR, "schema.json")
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            schema = json.load(f)
        print(f"\n[1] Dataset schema:")
        print(f"    Source: {schema.get('source','')}")
        print(f"    Total rows: {schema.get('total_rows',0):,}")
        print(f"    Snapshot date: {schema.get('date_snapshot','')}")
        print(f"    License: {schema.get('license','')}")
        print(f"    Columns: {schema.get('columns', [])}")
    
    # Load sample
    sample_path = os.path.join(RAW_DIR, "sample_5000.json")
    if not os.path.exists(sample_path):
        print("  NOTE: sample_5000.json not found. Reading from parquet requires pyarrow.")
        return

    rows = []
    with open(sample_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"\n[2] Sample: {len(rows)} rows loaded")
    
    if rows:
        from collections import Counter
        
        # Data type distribution
        data_types = Counter(r.get('dataType','') for r in rows)
        print(f"\n[3] Top data types collected:")
        for dt, cnt in data_types.most_common(10):
            print(f"    {dt}: {cnt}")
        
        # Genre distribution  
        genres = Counter(r.get('genreId','') for r in rows)
        print(f"\n[4] Top app genres:")
        for genre, cnt in genres.most_common(5):
            print(f"    {genre}: {cnt}")
        
        # Purpose distribution
        purposes = Counter(r.get('purpose','') for r in rows)
        print(f"\n[5] Top data collection purposes:")
        for p, cnt in purposes.most_common(5):
            print(f"    {p}: {cnt}")
        
        print(f"\n[6] Sample row:")
        print(f"    {json.dumps(rows[0], indent=4)}")

    print(f"\n[7] Full dataset: data/raw/play_data_safety/all_data_dsc_05_30.brotli")
    print(f"    HuggingFace: https://huggingface.co/datasets/WIPI/GoogleDataSafety")
    print(f"    Note: Use pyarrow to read parquet; batch iteration recommended (12.97M rows)")
    print("=" * 60)

if __name__ == "__main__":
    main()
