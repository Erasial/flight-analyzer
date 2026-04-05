import requests
import sys
import os
import time

def test_api():
    base_url = "http://localhost:8000"
    analyze_url = f"{base_url}/analyze"
    
    # Look for .BIN files in data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    bin_files = [f for f in os.listdir(data_dir) if f.endswith('.BIN')]
    
    if not bin_files:
        print("No .BIN files found in data/ folder.")
        return

    test_file = os.path.join(data_dir, bin_files[0])
    print(f"Testing API with file: {test_file}")

    # Step 1: Upload file and get result_id
    with open(test_file, 'rb') as f:
        files = {'file': (bin_files[0], f, 'application/octet-stream')}
        try:
            print("Step 1: Uploading file...")
            response = requests.post(analyze_url, files=files)
            if response.status_code == 200:
                upload_data = response.json()
                result_id = upload_data.get('result_id')
                print(f"Success! Received result_id: {result_id}")
                
                # Step 2: Retrieve results using result_id
                print(f"Step 2: Retrieving results for ID: {result_id}")
                result_url = f"{base_url}/results/{result_id}"
                result_response = requests.get(result_url)
                
                if result_response.status_code == 200:
                    data = result_response.json()
                    print("Successfully retrieved data!")
                    print(f"Filename: {data['filename']}")
                    print(f"Chart points count: {len(data['chart_points'])}")
                    print(f"Table preview rows: {len(data['table_preview'])}")
                    print("Metrics:")
                    for k, v in data['metrics'].items():
                        print(f"  {k}: {v}")
                else:
                    print(f"Error retrieving results: {result_response.status_code}")
                    print(result_response.text)
            else:
                print(f"Upload error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"An error occurred during API request: {e}")

if __name__ == "__main__":
    test_api()
