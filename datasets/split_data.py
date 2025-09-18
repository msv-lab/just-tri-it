import json
import os
import subprocess
from typing import List, Dict, Any


def extract_all_ids_from_json(json_file_path: str) -> list:
    # 1. Read the contents of the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)  # Load JSON data, which may be a dictionary or list
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {json_file_path} is not a valid JSON format")
        return []

    all_ids = []

    # 2. Define a recursive function to traverse all elements and extract ids
    def traverse_and_extract_id(data):
        # If current element is a dictionary: check for "id", add if present; then traverse dictionary values
        if isinstance(data, dict):
            # Extract "id" from current dictionary (if exists)
            if "id" in data:
                all_ids.append(data["id"])
            # Recursively traverse all values in the dictionary (may contain nested dictionaries/lists)
            for value in data.values():
                traverse_and_extract_id(value)
        # If current element is a list/tuple: traverse each element and process recursively
        elif isinstance(data, (list, tuple)):
            for item in data:
                traverse_and_extract_id(item)
        # Other types (e.g., strings, numbers, etc.): no processing needed

    # 3. Start traversal
    traverse_and_extract_id(json_data)

    # 4. Return the result
    return all_ids


def check_and_preprocess(id_value: str, output_dir: str = "./lcb_try") -> List[Dict[str, Any]]:
    """
    Check if the JSON file corresponding to the specified ID exists, and execute the preprocessing command if it does not

    Parameters:
        id_value: The ID value to check
        output_dir: Directory where the output files will be stored

    Returns:
        The loaded JSON data if successful, None otherwise
    """
    os.makedirs(output_dir, exist_ok=True)

    # Target file path
    target_path = f"{output_dir}/{id_value}.json"

    # Check if the file exists
    if not os.path.exists(target_path):
        print(f"File {target_path} does not exist, starting to execute preprocessing command...")

        # Build the command to be executed
        command = [
            "uv", "run", "preprocess_dataset",
            "--decompress",
            "--dataset", "lcb_part6.json",
            "--task", id_value,
            "--output", target_path
        ]

        try:
            # Execute the command and wait for completion
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Command executed successfully, output file: {target_path}")
            print("Command output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command execution failed, error code: {e.returncode}")
            print("Error output:", e.stderr)
            return None
    else:
        print(f"File {target_path} already exists, no processing needed")

    # Load and return the processed data
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {target_path}: {e}")
        return []


def save_data_batch(data_batch: List[Dict[str, Any]], batch_index: int, output_dir: str = "./lcb_try_batches"):
    """
    Save a batch of data to a JSON file

    Parameters:
        data_batch: List of JSON data to save
        batch_index: Index of the batch (used for naming the output file)
        output_dir: Directory where the batch files will be stored
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create output file path
    output_path = f"{output_dir}/lcb_batch_{batch_index}.json"

    # Save the batch data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_batch, f, ensure_ascii=False, indent=2)

    print(f"Saved batch {batch_index} to {output_path} with {len(data_batch)} items")


def process_all_ids(ids: List[str], batch_size: int = 10):
    """
    Process all IDs in batches

    Parameters:
        ids: List of IDs to process
        batch_size: Number of items to process before saving a batch
    """
    all_data = []
    current_batch = []
    batch_count = 0

    for i, id_val in enumerate(ids):
        print(f"Processing ID {i + 1}/{len(ids)}: {id_val}")

        # Process the current ID
        data = check_and_preprocess(id_val)

        if data is not None:
            current_batch.append(data[0])

            # If we've reached the batch size, save the batch
            if len(current_batch) >= batch_size:
                save_data_batch(current_batch, batch_count)
                all_data.extend(current_batch)
                current_batch = []
                batch_count += 1

    # Save any remaining items in the last batch
    if current_batch:
        save_data_batch(current_batch, batch_count)
        all_data.extend(current_batch)

    return all_data


# Extract all IDs from the JSON file
extracted_ids = extract_all_ids_from_json("./lcb_part6.json")
print(f"Found {len(extracted_ids)} IDs to process")

# Process all IDs in batches
all_processed_data = process_all_ids(extracted_ids, batch_size=10)
print(f"Processed {len(all_processed_data)} items in total")