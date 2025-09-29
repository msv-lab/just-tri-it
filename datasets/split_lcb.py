import json
import os


def simple_split_json(input_file, output_dir="output"):
    """
    Simplified JSON file splitting
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    for item in data:
        if isinstance(item, dict) and 'id' in item:
            filename = f"{item['id']}.json"
            output_path = os.path.join(output_dir, filename)

            # Put data into an array
            output_data = [item]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"Created: {filename}")


# Usage example
if __name__ == "__main__":
    simple_split_json("lcb_part6.json", "output_files")