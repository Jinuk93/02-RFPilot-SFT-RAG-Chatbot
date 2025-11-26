from unsloth import FastLanguageModel
import torch
import os
import glob

# ==========================================
# [Smart Export Script] 최신 체크포인트 자동 감지
# ==========================================

print(">>> [System] GGUF 변환 작업을 시작합니다...")

# 1. 최신 체크포인트 폴더 찾기 (핵심!)
output_dir = "outputs_final"

if not os.path.exists(output_dir):
    print(f">>> [Error] '{output_dir}' 폴더가 없습니다!")
    exit()

# checkpoint- 숫자 폴더들을 다 찾아서 숫자가 제일 큰 놈을 고름
subfolders = [f.path for f in os.scandir(output_dir) if f.is_dir() and "checkpoint" in f.name]

if not subfolders:
    print(">>> [Error] 체크포인트 폴더를 찾을 수 없습니다!")
    exit()

# 숫자로 정렬해서 가장 마지막 것 선택 (예: checkpoint-3171)
latest_checkpoint = max(subfolders, key=lambda x: int(x.split('-')[-1]))

print(f">>> [Found] 가장 학습이 잘 된 모델을 찾았습니다: {latest_checkpoint}")
print(">>> [Model] 모델 로드 중... (xFormers 경고는 무시하세요)")

# 2. 모델 로드 (정확한 경로 입력)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = latest_checkpoint, # <--- 자동으로 찾은 경로
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 3. GGUF 변환
print(f">>> [Convert] '{latest_checkpoint}' -> GGUF 변환 시작 (5~10분 소요)")

# q4_k_m: 용량/성능 밸런스형
model.save_pretrained_gguf("BiddinMate_Model", tokenizer, quantization_method = "q4_k_m")

print(">>> [Success] 변환 완료!")
print(f">>> 'BiddinMate_Model' 폴더 안에 .gguf 파일이 생성되었습니다.")
