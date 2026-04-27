"""
Load Princeton-Leuven Privacy Policy Corpus sample and print stats.
Source: https://github.com/citp/privacy-policy-historical (master branch)
Full SQLite: https://privacypolicies.cs.princeton.edu/data (3.1GB compressed, skipped)
We sampled 5,000 markdown files from the 130,620-file repo.
"""
import json
import os

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/princeton_ppc"))

def main():
    print("=" * 60)
    print("Princeton-Leuven Privacy Policy Corpus Loader")
    print("=" * 60)

    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("  ERROR: manifest.json not found. Run M1 acquisition first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n[1] Manifest: {len(manifest)} policies")
    
    total_chars = sum(m['char_count'] for m in manifest)
    total_bytes = sum(m['file_size'] for m in manifest)
    avg_chars = total_chars / len(manifest) if manifest else 0
    
    print(f"    Total size on disk: {total_bytes / 1e6:.1f} MB")
    print(f"    Total characters: {total_chars:,}")
    print(f"    Avg characters per policy: {avg_chars:,.0f}")

    # Distribution of sizes
    small = sum(1 for m in manifest if m['char_count'] < 1000)
    medium = sum(1 for m in manifest if 1000 <= m['char_count'] < 10000)
    large = sum(1 for m in manifest if m['char_count'] >= 10000)
    print(f"    Size buckets: <1k={small}, 1k-10k={medium}, >10k={large}")

    print(f"\n[2] Sampling 5 policies:")
    import random
    random.seed(0)
    for m in random.sample(manifest, min(5, len(manifest))):
        print(f"    domain={m['domain']} | chars={m['char_count']} | size={m['file_size']}B")
        print(f"      snippet: {m['snippet_first_500_chars'][:80]}...")

    print(f"\n[3] Full corpus stats:")
    print(f"    Total files in repo: 130,620")
    print(f"    Sampled: {len(manifest)}")
    print(f"    Source: https://github.com/citp/privacy-policy-historical")
    print(f"    License: unknown (academic use)")
    print("=" * 60)

if __name__ == "__main__":
    main()
