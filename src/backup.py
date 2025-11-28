from minio import Minio
from minio.error import S3Error
from openai import OpenAI
import os

# Initialize MinIO client
minio_client = Minio(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

# Initialize OpenAI client
openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")

buckets = minio_client.list_buckets()
for bucket in buckets:
    print(bucket.name)

bucket_name = "budget"

try:
    objects = minio_client.list_objects(bucket_name, recursive=True)

    print(f"\nReading files from bucket: {bucket_name}")
    print("-" * 50)

    for obj in objects:
        print(f"\nFile: {obj.object_name}")
        print(f"Size: {obj.size} bytes")
        print(f"Last Modified: {obj.last_modified}")

        # Get the object data
        try:
            response = minio_client.get_object(bucket_name, obj.object_name)
            data = response.read()

            # Try to decode as text (if it's a text file)
            try:
                content = data.decode("utf-8")
                print(f"Content:\n{content[:500]}")  # Print first 500 chars
                if len(content) > 500:
                    print(f"... (truncated, total length: {len(content)} characters)")

                # Pass content to OpenAI LLM
                print("\n--- Sending to OpenAI LLM ---")
                llm_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that analyzes document content.",
                        },
                        {
                            "role": "user",
                            "content": f"Please summarize the following document from {obj.object_name}:\n\n{content}",
                        },
                    ],
                    max_tokens=500,
                )
                print(f"LLM Summary:\n{llm_response.choices[0].message.content}")
                raise RuntimeError(
                    "Debug stop after first file"
                )  # Remove or comment this line to process all files

            except UnicodeDecodeError:
                print("Binary file (cannot display as text)")

            response.close()
            response.release_conn()
        except (S3Error, OSError) as e:
            print(f"Error reading file {obj.object_name}: {e}")

        print("-" * 50)

except S3Error as e:
    print(f"Error accessing bucket '{bucket_name}': {e}")
