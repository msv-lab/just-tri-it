import json
import os
import subprocess


def extract_all_ids_from_json(json_file_path: str) -> list:
    """
    Read a JSON file, traverse all dictionaries and extract the "id" field, return a list of all ids

    Args:
        json_file_path: Path to the JSON file (e.g., "./data.json")

    Returns:
        A list containing all extracted "id" values (dictionaries without the "id" field will be skipped)
    """
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


def check_and_preprocess(id_value):
    """
    Check if the JSON file corresponding to the specified ID exists, and execute the preprocessing command if it does not

    Parameters:
        id_value: The ID value to check
    """
    # Target file path
    target_path = f"./lcb_try/{id_value}.json"

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
    else:
        print(f"File {target_path} already exists, no processing needed")


extracted_ids = extract_all_ids_from_json("./lcb_part6.json")
for id_val in extracted_ids[:20]:
    check_and_preprocess(id_val)