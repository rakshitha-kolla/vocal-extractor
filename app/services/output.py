import os
import json
from datetime import datetime


def save_output(filename: str, result: dict, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return output_filename