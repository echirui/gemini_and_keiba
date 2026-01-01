import os
import time
import requests


def download_race_pages():
    """
    Reads race IDs from race_id.txt and downloads the corresponding
    race pages from db.netkeiba.com.
    """
    race_id_file = "race_id.txt"
    output_dir = "data"
    base_url = "https://db.netkeiba.com/race/"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(race_id_file, "r") as f:
            race_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {race_id_file} not found.")
        return

    total_races = len(race_ids)
    print(f"Found {total_races} race IDs to download.")

    for i, race_id in enumerate(race_ids):
        file_path = os.path.join(output_dir, f"{race_id}.html")

        if os.path.exists(file_path):
            print(f"({i + 1}/{total_races}) Skipping {race_id}: File already exists.")
            continue

        url = f"{base_url}{race_id}"

        try:
            print(f"({i + 1}/{total_races}) Downloading {race_id} from {url}...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes

            # The website uses 'EUC-JP' encoding.
            response.encoding = "EUC-JP"

            with open(file_path, "w", encoding="utf-8") as html_file:
                html_file.write(response.text)

            print(f"Successfully saved to {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {race_id}: {e}")

        # Wait for 1 second between requests to be polite to the server
        time.sleep(1)


if __name__ == "__main__":
    download_race_pages()
