import json
from collections import Counter

INPUT_FILE = "outputs/hallucination_results.json"


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    total = len(data)
    hallucinated_small = 0
    hallucinated_large = 0
    hallucinated_either = 0

    small_counter = Counter()
    large_counter = Counter()

    for image_id, result in data.items():
        hs = result["hallucinated_small"]
        hl = result["hallucinated_large"]

        if hs:
            hallucinated_small += 1
            small_counter[result["small_removed"]] += 1
        if hl:
            hallucinated_large += 1
            large_counter[result["large_removed"]] += 1
        if hs or hl:
            hallucinated_either += 1

    print("Hallucination Report")
    print("=" * 40)
    print(f"Total Images Evaluated       : {total}")
    print(
        f"Small Object Hallucinations : {hallucinated_small} ({hallucinated_small / total:.2%})"
    )
    print(
        f"Large Object Hallucinations : {hallucinated_large} ({hallucinated_large / total:.2%})"
    )
    print(
        f"Any Hallucination Present   : {hallucinated_either} ({hallucinated_either / total:.2%})"
    )

    print("\n🔁 Top 10 Hallucinated Small Objects:")
    for obj, count in small_counter.most_common(10):
        print(f"  - {obj}: {count} times")

    print("\n🔁 Top 10 Hallucinated Large Objects:")
    for obj, count in large_counter.most_common(10):
        print(f"  - {obj}: {count} times")


if __name__ == "__main__":
    main()
