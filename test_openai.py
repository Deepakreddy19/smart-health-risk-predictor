from openai import OpenAI

# ⚠️ Replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key
client = OpenAI(api_key="YOUR_API_KEY_HERE")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello, can you confirm my setup is working?"}
    ]
)

print(response.choices[0].message.content)
