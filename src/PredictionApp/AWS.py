import boto3
import json


def main():
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")
    prompt = "I lost my debit card can you help me"
    response = ""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    try:
       response = bedrock_client.invoke_model(modelId = 'arn:aws:bedrock:us-east-2:524873268749:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0', body = json.dumps(payload))
    except Exception as e:
        print(f"ERROR: {e} ")
        return

    response_body = json.loads(response["body"].read())
    text = response_body["content"][0]["text"]

    print("Generated Output: ", text)

if __name__ == "__main__":
    main()

