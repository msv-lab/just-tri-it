import random
import json
import os

def select_random_ids_from_json(json_file_path, n, output_file):
    """
    Read data from the specified JSON file, randomly select n entries, 
    and write their ids to a text file
    
    Parameters:
    json_file_path: Path to the JSON file
    n: Number of entries to select
    output_file: Path for the output text file
    """
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data format
        if not isinstance(data, list):
            raise ValueError("JSON file content must be a list")
            
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("Each element in the list must be a dictionary containing the 'id' key")
                
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not valid JSON format")
        return []
    except Exception as e:
        print(f"Error occurred while reading file: {str(e)}")
        return []
    
    # Check if n is greater than total number of data entries
    if n > len(data):
        print(f"Warning: Requested selection count ({n}) is greater than total data entries ({len(data)}), will return all ids")
        n = len(data)
    elif n <= 0:
        print(f"Warning: Selection count ({n}) must be a positive number, returning 0 results")
        return []
    
    # Randomly select n entries
    selected_items = random.sample(data, n)
    
    # Extract ids of selected entries
    selected_ids = [item["id"] for item in selected_items]
    
    # Write ids to text file
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for id in selected_ids:
                f.write(id + '\n')
        
        print(f"Successfully wrote {len(selected_ids)} randomly selected ids to {output_file}")
        return selected_ids
    except Exception as e:
        print(f"Error occurred while writing file: {str(e)}")
        return []

if __name__ == "__main__":
    # JSON file path (please modify to your actual file path)
    json_file_path = "lcb_part6.json"  # Replace with your JSON file path
    
    # Number of items to select (can be modified as needed)
    n = 30
    
    # Output file path
    output_file = "lcb_random30.txt"
    
    # Execute selection and write to file
    selected_ids = select_random_ids_from_json(json_file_path, n, output_file)

