import os
import shutil
import hashlib
from collections import Counter, defaultdict

from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = r"C:\Users\USER\Desktop\cnn_image\mammals"
CLEAN_DIR = r"C:\Users\USER\Desktop\cnn_image\mammals_clean"
OUTPUT_DIR = r"C:\Users\USER\Desktop\cnn_image\eda_results"

EXPECTED_CLASSES = 44

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DARK_THRESHOLD = 40
BRIGHT_THRESHOLD = 220


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):
    """Return all image files inside a folder."""
    files = []

    if not os.path.exists(folder):
        return files

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(VALID_EXTENSIONS):
                files.append(os.path.join(root, filename))

    return files


def calculate_md5(filepath):
    """Calculate MD5 hash of an image file."""
    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


def get_class_counts(folder):
    """Count images in each class folder."""
    counts = {}

    if not os.path.exists(folder):
        return counts

    for class_name in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)

        if os.path.isdir(class_path):
            images = get_image_files(class_path)
            counts[class_name] = len(images)

    return counts


def save_class_distribution(counts, filename):
    """Save class distribution as CSV."""
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("class,image_count\n")

        for class_name, count in sorted(counts.items()):
            f.write(f"{class_name},{count}\n")

    return filepath


def plot_class_distribution(counts, title, filename):
    """Create class distribution graph."""
    if not counts:
        return

    classes = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(16, 8))

    plt.bar(classes, values)

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    plt.xticks(rotation=90)

    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(filepath, dpi=200)

    plt.close()


# ============================================================
# 1. CHECK SOURCE DATASET
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL MAMMAL DATASET EDA")
print("=" * 70)

print(f"\nOriginal dataset:")
print(SOURCE_DIR)

print(f"\nClean dataset:")
print(CLEAN_DIR)

print(f"\nEDA results:")
print(OUTPUT_DIR)


if not os.path.exists(SOURCE_DIR):
    print("\nERROR: Source dataset does not exist.")
    print(SOURCE_DIR)
    exit()


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = get_class_counts(SOURCE_DIR)

print(f"\nNumber of classes: {len(class_counts)}")
print(f"Total images: {sum(class_counts.values())}")

print("\nImages per class:")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name:<25} {count}")


# ============================================================
# 3. CHECK EXPECTED CLASS COUNT
# ============================================================

if len(class_counts) == EXPECTED_CLASSES:
    print(f"\n✓ {EXPECTED_CLASSES} classes detected")
else:
    print(
        f"\n⚠ WARNING: Expected {EXPECTED_CLASSES} classes "
        f"but found {len(class_counts)}"
    )


# ============================================================
# 4. CLASS STATISTICS
# ============================================================

counts = list(class_counts.values())

if counts:

    minimum = min(counts)
    maximum = max(counts)
    average = sum(counts) / len(counts)

    sorted_counts = sorted(counts)

    middle = len(sorted_counts) // 2

    if len(sorted_counts) % 2 == 0:
        median = (
            sorted_counts[middle - 1] +
            sorted_counts[middle]
        ) / 2
    else:
        median = sorted_counts[middle]

    mean = average

    variance = sum(
        (x - mean) ** 2 for x in counts
    ) / len(counts)

    std = variance ** 0.5

    print("\nClass statistics:")
    print(f"Minimum images : {minimum}")
    print(f"Maximum images : {maximum}")
    print(f"Average images : {average:.2f}")
    print(f"Median images  : {median}")
    print(f"Std deviation  : {std:.2f}")

    print("\n")


# ============================================================
# 5. SAVE CLASS DISTRIBUTION
# ============================================================

csv_path = save_class_distribution(
    class_counts,
    "class_distribution.csv"
)

print(f"Class distribution saved:")
print(csv_path)


# ============================================================
# 6. CLASS DISTRIBUTION GRAPH
# ============================================================

plot_class_distribution(
    class_counts,
    "Original Dataset Class Distribution",
    "class_distribution.png"
)

print("\nClass distribution graph saved.")


# ============================================================
# 7. CHECK CORRUPT IMAGES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING CORRUPT IMAGES")
print("=" * 70)

all_images = get_image_files(SOURCE_DIR)

corrupt_images = []

for index, filepath in enumerate(all_images, start=1):

    try:
        with Image.open(filepath) as img:
            img.verify()

    except Exception:
        corrupt_images.append(filepath)

print(f"\nTotal images checked: {len(all_images)}")
print(f"Corrupt images: {len(corrupt_images)}")

corrupt_report = os.path.join(
    OUTPUT_DIR,
    "corrupt_images.txt"
)

with open(corrupt_report, "w", encoding="utf-8") as f:

    for filepath in corrupt_images:
        f.write(filepath + "\n")

if len(corrupt_images) == 0:
    print("✓ No corrupt images found.")
else:
    print(f"⚠ Corrupt image report saved:")
    print(corrupt_report)


# ============================================================
# 8. CHECK IMAGE DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING IMAGE DIMENSIONS")
print("=" * 70)

dimension_counter = Counter()

for filepath in all_images:

    try:
        with Image.open(filepath) as img:
            dimension_counter[img.size] += 1

    except Exception:
        pass


print("\nImage dimensions:")

for dimension, count in dimension_counter.most_common():
    print(f"{dimension}: {count}")


dimension_report = os.path.join(
    OUTPUT_DIR,
    "image_dimensions.txt"
)

with open(dimension_report, "w", encoding="utf-8") as f:

    for dimension, count in dimension_counter.most_common():
        f.write(f"{dimension}: {count}\n")


# ============================================================
# 9. DARK AND BRIGHT IMAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING VERY DARK / VERY BRIGHT IMAGES")
print("=" * 70)

dark_images = []
bright_images = []

for filepath in all_images:

    try:

        with Image.open(filepath).convert("L") as img:

            pixels = list(img.getdata())

            if len(pixels) == 0:
                continue

            average_brightness = sum(pixels) / len(pixels)

            if average_brightness < DARK_THRESHOLD:
                dark_images.append(filepath)

            elif average_brightness > BRIGHT_THRESHOLD:
                bright_images.append(filepath)

    except Exception:
        pass


print(f"\nVery dark images : {len(dark_images)}")
print(f"Very bright images: {len(bright_images)}")


brightness_report = os.path.join(
    OUTPUT_DIR,
    "brightness_report.txt"
)

with open(brightness_report, "w", encoding="utf-8") as f:

    f.write("VERY DARK IMAGES\n")
    f.write("=" * 50 + "\n")

    for filepath in dark_images:
        f.write(filepath + "\n")

    f.write("\n\nVERY BRIGHT IMAGES\n")
    f.write("=" * 50 + "\n")

    for filepath in bright_images:
        f.write(filepath + "\n")


# ============================================================
# 10. EXACT DUPLICATE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("CHECKING EXACT DUPLICATES")
print("=" * 70)

hash_groups = defaultdict(list)

for index, filepath in enumerate(all_images, start=1):

    try:

        file_hash = calculate_md5(filepath)

        hash_groups[file_hash].append(filepath)

    except Exception:
        pass


duplicate_groups = {
    file_hash: paths
    for file_hash, paths in hash_groups.items()
    if len(paths) > 1
}


print(f"\nDuplicate groups: {len(duplicate_groups)}")


duplicate_report = os.path.join(
    OUTPUT_DIR,
    "duplicate_report.txt"
)

same_class_groups = []
cross_class_groups = []

for file_hash, paths in duplicate_groups.items():

    classes = set(
        os.path.basename(os.path.dirname(path))
        for path in paths
    )

    if len(classes) == 1:
        same_class_groups.append(paths)
    else:
        cross_class_groups.append(paths)


print(f"Same-class duplicate groups : {len(same_class_groups)}")
print(f"Cross-class duplicate groups: {len(cross_class_groups)}")


redundant_same_class = sum(
    len(paths) - 1
    for paths in same_class_groups
)

print(
    f"Redundant same-class images: "
    f"{redundant_same_class}"
)


with open(duplicate_report, "w", encoding="utf-8") as f:

    f.write("EXACT DUPLICATE REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Duplicate groups: {len(duplicate_groups)}\n"
    )

    f.write(
        f"Same-class groups: {len(same_class_groups)}\n"
    )

    f.write(
        f"Cross-class groups: {len(cross_class_groups)}\n"
    )

    f.write(
        f"Redundant same-class images: "
        f"{redundant_same_class}\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("SAME-CLASS DUPLICATES\n")
    f.write("=" * 70 + "\n\n")

    for group in same_class_groups:

        for path in group:
            f.write(path + "\n")

        f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("CROSS-CLASS DUPLICATES\n")
    f.write("=" * 70 + "\n\n")

    for group in cross_class_groups:

        for path in group:
            f.write(path + "\n")

        f.write("\n")


print(f"\nDuplicate report saved:")
print(duplicate_report)


# ============================================================
# 11. CREATE CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING CLEAN DATASET")
print("=" * 70)


if os.path.exists(CLEAN_DIR):

    print("\nClean dataset already exists.")
    print("It will NOT be overwritten.")

else:

    print("\nCreating clean dataset...")

    os.makedirs(CLEAN_DIR)

    copied_count = 0
    removed_count = 0

    # Create class folders
    for class_name in class_counts:

        os.makedirs(
            os.path.join(CLEAN_DIR, class_name),
            exist_ok=True
        )

    # Keep the first image of each same-class duplicate group
    files_to_remove = set()

    for group in same_class_groups:

        for duplicate_path in group[1:]:
            files_to_remove.add(
                os.path.abspath(duplicate_path)
            )

    for filepath in all_images:

        if os.path.abspath(filepath) in files_to_remove:

            removed_count += 1
            continue

        class_name = os.path.basename(
            os.path.dirname(filepath)
        )

        destination = os.path.join(
            CLEAN_DIR,
            class_name,
            os.path.basename(filepath)
        )

        shutil.copy2(filepath, destination)

        copied_count += 1

    print("\nClean dataset created.")

    print(f"Copied images : {copied_count}")
    print(f"Removed images: {removed_count}")


# ============================================================
# 12. VERIFY CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING CLEAN DATASET")
print("=" * 70)

clean_class_counts = get_class_counts(CLEAN_DIR)

clean_total = sum(clean_class_counts.values())

print(f"\nClean dataset classes: {len(clean_class_counts)}")
print(f"Clean dataset images : {clean_total}")

print("\nClean class distribution:")

for class_name, count in sorted(clean_class_counts.items()):

    print(f"{class_name:<25} {count}")


# ============================================================
# 13. VERIFY 44 CLASSES
# ============================================================

if len(clean_class_counts) == 44:

    print("\n✓ 44 classes detected in clean dataset.")

else:

    print(
        f"\n⚠ WARNING: Expected 44 classes "
        f"but found {len(clean_class_counts)}."
    )


# ============================================================
# 14. CHECK CROSS-CLASS DUPLICATES IN CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("CHECKING CROSS-CLASS DUPLICATES IN CLEAN DATASET")
print("=" * 70)

clean_images = get_image_files(CLEAN_DIR)

clean_hash_groups = defaultdict(list)

for filepath in clean_images:

    try:

        file_hash = calculate_md5(filepath)

        clean_hash_groups[file_hash].append(filepath)

    except Exception:
        pass


clean_duplicate_groups = {
    file_hash: paths
    for file_hash, paths in clean_hash_groups.items()
    if len(paths) > 1
}


clean_cross_class_groups = []

for file_hash, paths in clean_duplicate_groups.items():

    classes = set(
        os.path.basename(os.path.dirname(path))
        for path in paths
    )

    if len(classes) > 1:
        clean_cross_class_groups.append(paths)


print(
    f"\nCross-class duplicate groups in clean dataset: "
    f"{len(clean_cross_class_groups)}"
)


clean_cross_report = os.path.join(
    OUTPUT_DIR,
    "clean_cross_class_duplicates.txt"
)

with open(clean_cross_report, "w", encoding="utf-8") as f:

    for group in clean_cross_class_groups:

        for path in group:
            f.write(path + "\n")

        f.write("\n")


if len(clean_cross_class_groups) == 0:

    print("✓ No cross-class duplicates found.")

else:

    print(
        "⚠ Cross-class duplicates found."
    )

    print(
        f"Report saved: {clean_cross_report}"
    )


# ============================================================
# 15. SAVE CLEAN CLASS DISTRIBUTION
# ============================================================

clean_csv = save_class_distribution(
    clean_class_counts,
    "clean_class_distribution.csv"
)

print("\nClean class distribution saved:")
print(clean_csv)


# ============================================================
# 16. CLEAN DATASET GRAPH
# ============================================================

plot_class_distribution(
    clean_class_counts,
    "Clean Dataset Class Distribution",
    "clean_class_distribution.png"
)

print("Clean dataset graph saved.")


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL EDA SUMMARY")
print("=" * 70)

print(f"\nOriginal dataset:")
print(f"  Classes          : {len(class_counts)}")
print(f"  Images           : {len(all_images)}")

print("\nClean dataset:")
print(f"  Classes          : {len(clean_class_counts)}")
print(f"  Images           : {clean_total}")

print("\nCleaning:")
print(
    f"  Same-class duplicates removed: "
    f"{redundant_same_class}"
)

print(
    f"  Cross-class duplicates in clean: "
    f"{len(clean_cross_class_groups)}"
)

print(f"\nCorrupt images:")
print(f"  {len(corrupt_images)}")

print(f"\nVery dark images:")
print(f"  {len(dark_images)}")

print(f"\nVery bright images:")
print(f"  {len(bright_images)}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("\nYour FINAL dataset for training is:")
print(CLEAN_DIR)

print("\nNumber of classes: 44")
print("Seal is NOT included.")