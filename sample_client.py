import requests

# Define the URL of the endpoint
url = "http://localhost:8000/process_videos"

files = {
    "input_videos": ("1", open(r"D:\Projects\media\LoveTheWayYouLie.mp4", "rb")),
}

# Make the POST request
response = requests.post(url, files=files)

# Print the response
print(response.json())
