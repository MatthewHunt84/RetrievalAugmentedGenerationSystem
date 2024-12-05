# # test_openai.py
# from dotenv import load_dotenv
# load_dotenv()
#
# from openai import OpenAI
# client = OpenAI()
#
# try:
#     response = client.embeddings.create(
#         model="text-embedding-3-large",
#         input="Hello, world!"
#     )
#     print("Connection successful!")
# except Exception as e:
#     print(f"Connection failed: {str(e)}")