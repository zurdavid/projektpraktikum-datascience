from pathlib import Path

import httpx
import json
#  from generate_test_data import generate_test_data

test_files = [
    "camera_values_missing-b19f3361-59e3-4644-9299-f3484fdfd007.json",
    "excluded_discount_fraud-2835d96f-f2c4-4ec9-befc-dba05873a5d6.json",
    "fraud-1be63d50-7cff-4ed7-922e-68929b99036f.json",
    "no_lines-cce5317f-e4b6-4890-b776-fd9838730ae2.json",
    "non_fraud-7ba24ec9-37fc-4736-a04e-2d1898b2ebef.json",
    "unscanned_products-77578667-8427-46de-8d08-4008d551d770.json",
    "unscanned_products_product_id_missing-b5450bf4-a819-4e95-8de7-9b29220e7281.json",
    "valid.json",
]

path = Path("./tests/json")


def make_api_call(payload):
    url = "http://localhost:8080/fraud-prediction"
    headers = {"Content-Type": "application/json"}

    try:
        response = httpx.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        print(response)
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    i = 2
    with open(path / test_files[i], "r") as f:
        payload = json.load(f)
    # payload = generate_test_data()
    result = make_api_call(payload)
    if result:
        print(json.dumps(payload, indent=4, ensure_ascii=False))
        print("\nAPI Response:")
        result = json.dumps(result, indent=4, ensure_ascii=False)
    else:
        result = "No response received or an error occurred."
    print(result)
