import requests
import os

# This is the direct link to the raw binary
url = "https://github.com/chenosaurus/poker-evaluator/raw/master/data/HandRanks.dat"
dest = "C:\\Users\\422mi\\Pulselib\\environments\\Poker\\HandRanks.dat" 

# Ensure the directory exists
os.makedirs(os.path.dirname(dest), exist_ok=True)

print(f"Downloading clean HandRanks.dat...")
response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print(f"✅ Success! File saved to {dest}")
    
    # Verify the size immediately
    size = os.path.getsize(dest)
    expected_size = 129404512
    if size == expected_size:
        print("✅ INTEGRITY CHECK PASSED: File size is exactly 129,404,512 bytes.")
    else:
        print(f"❌ FAIL: Size is {size} bytes (Expected {expected_size}).")
else:
    print(f"❌ Download failed with code {response.status_code}")