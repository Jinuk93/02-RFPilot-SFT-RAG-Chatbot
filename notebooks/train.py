import torch
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import os
import shutil
import time

# ==========================================
# [FINAL SCRIPT] Running on Terminal
# ==========================================

print(">>> [System] 스크립트 시작. 라이브러리 로딩 완료.")

# 1. WandB 찌꺼기 폴더 강제 삭제
if os.path.exists("wandb"):
    try:
        shutil.rmtree("wandb")
        print(">>> [System] 기존 WandB 캐시 삭제 완료")
    except:
        pass

# 2. 복구 모드 점검
output_dir = "outputs_final"
last_checkpoint = None

if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        print(f">>> [Resume] 이전 학습 기록 발견: {last_checkpoint}")
    else:
        print(">>> [Start] 새로운 학습 시작")

# 3. WandB 설정
try:
    wandb.finish()
except:
    pass

unique_id = f"run_{int(time.time())}"

wandb.init(
    entity="hambur1203-project",
    project="BiddinMate_Production_SFT",
    name="Llama3-8B-Final-3Epochs",
    id=unique_id,
    resume="allow"
)

# 4. 모델 로드 (0번 GPU 강제 지정)
print(">>> [Model] Llama-3 로드 중...")
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "beomi/Llama-3-Open-Ko-8B",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    device_map = {"": 0} # 핵심: GPU 0번 고정
)

# 5. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 6. 데이터셋 로드
print(">>> [Data] 데이터셋 로드 중...")
dataset = load_dataset("json", data_files="sft_train_llama.jsonl", split="train")

# 7. 학습 설정
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        warmup_steps = 100,
        learning_rate = 2e-4,
        report_to = "wandb",
        run_name = "Llama3-8B-Final-3Epochs",
        logging_steps = 1,
        save_strategy = "epoch",
        output_dir = output_dir,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        seed = 3407,
    ),
)

# 8. 실행
print(">>> [Train] 학습 시작! (WandB를 확인하세요)")
if last_checkpoint:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
