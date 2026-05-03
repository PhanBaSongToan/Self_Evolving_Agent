from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import os

# Resolve project root = parent of this script's directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Đăng nhập bằng Token
HF_TOKEN = "HF_TOKEN"
login(token=HF_TOKEN)

# 2. Tải dataset
print("Đang tải dữ liệu, vui lòng chờ...")
dataset = load_dataset("lmsys/chatbot_arena_conversations")

# 3. Chuyển thành Pandas DataFrame
df = dataset["train"].to_pandas()

# 4. Trích xuất câu hỏi của người dùng
def extract_user_prompt(conversation_list):
    try:
        return conversation_list[0]['content']
    except:
        return ""

df['user_prompt'] = df['conversation_a'].apply(extract_user_prompt)

# 5. Lưu ra file CSV
out_dir = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'raw_arena_data.csv')
df.to_csv(out_path, index=False, encoding='utf-8')

print(f"Thành công! Đã lưu file tại: {out_path}")