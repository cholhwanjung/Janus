import json
import os
import pickle
import sys
import time
from io import BytesIO

import pandas as pd
import torch
import wandb

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


class InstructionTuningDataset(Dataset):
    def __init__(self, df):
        df = df.reset_index(drop=True)
        self.images = df["image_bytes"]
        self.caption = df["caption"]
        self.subcategory = df["subcategory"]
        self.category = df["category"]
        self.index = df.index
        self.sft_format = "<|User|>: <image_placeholder>\n\n{}\n\n<|Assistant|>: {}"

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        image = Image.open(BytesIO(self.images[idx]))
        caption = self.caption[idx]
        category = self.category[idx]

        if category == "TOP":
            instruction = "Describe the top garment in detail in the given image."
        elif category == "BOTTOM":
            instruction = "Describe the bottom garment in detail in the given image."

        prompt = self.sft_format.format(instruction, caption)

        return {"prompt": prompt, "image": [image]}


def collate_fn(batch):
    batch_dict = {key: [sample[key] for sample in batch] for key in batch[0]}
    
    prompts = batch_dict["prompt"]
    images = batch_dict["image"]
    
    # Process batch using vl_chat_processor
    prepare_batch_inputs = processor.process_batch(prompt_batch=prompts, images_batch=images)
    
    return prepare_batch_inputs


if __name__ == "__main__":
    wandb_project = "Janus Tuning"
    run_name = "test"

    train_dataset_path = "/data/charles/data/farfetch_full_250318.parquet"
    train_meta_path = "/data/charles/data/farfetch_meta_250314.json"
    result_path = "/data/charles/data/full_image_result_250321.json"

    with open(train_meta_path, "r") as file:
        train_meta = json.load(file)

    with open("/home/charles/VLM/Janus/notebook/category_map.pickle", "rb") as file:
        category_map = pickle.load(file)

    with open(result_path, "r") as file:
	    result = json.load(file)

    result_df = pd.DataFrame.from_dict([{"index": int(k), "caption": v} for k, v in result.items()]).set_index("index")

    category_map.pop(())
    subcategory_set = set(i[-1] for i in category_map.keys())


    base_save_dir = "/data/charles/ckpt"
    model_type = "JP 7b"
    scheme = "test instruction"

    batch_size = 2
    num_epochs = 2
    learning_rate = 1e-5

    torch.manual_seed(23)

    # Initialize model and processor
    model_path = "deepseek-ai/Janus-Pro-7B"
    peft_adapter_path = ""
    do_peft = True
    adapt = False
    enable_zero3 = False
    
    log_wandb = True

    model_save_dir = f"{base_save_dir}/{model_type}-{scheme}-lora {do_peft}-zero3 {enable_zero3}"

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir="/data/charles/huggingface/hub"
    ).to(torch.bfloat16)

    processor = VLChatProcessor.from_pretrained(
        model_path,
        cache_dir="/data/charles/huggingface/hub"
    )

    if do_peft:
        if adapt:
            print("adapter")
            model = PeftModel.from_pretrained(model, peft_adapter_path)
        else:
            print("no adapter")
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, config)

    train_data = pd.read_parquet(train_dataset_path, engine="pyarrow", columns=None)
    train_data["long_desc"] = train_data["code"].apply(lambda x: train_meta[x]["long_desc"])
    train_data["image_index"] = train_data["image_path"].apply(lambda x: int(x.split("_")[-1].split(".")[0]))
    train_data["subcategory"] = train_data["code"].apply(lambda x: train_meta[x]["subcategory"])
    train_data["category"] = train_data["code"].apply(lambda x: train_meta[x]["category"])

    main_images = train_data[train_data["image_index"]==1]
    main_images = main_images[main_images["subcategory"].isin(subcategory_set)].reset_index(drop=True)

    # suffle or not
    shuffled_images = main_images.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = shuffled_images[(shuffled_images.index.isin(result_df.index))&(shuffled_images["category"].isin(["TOP", "BOTTOM"]))]
    train_df = pd.merge(train_df, result_df, left_index=True, right_index=True).reset_index(drop=True)

    del train_data, main_images, shuffled_images

    # split and create dataset
    train_data, val_data = train_test_split(train_df, test_size=0.05, random_state=42)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    del train_df

    train_dataset = InstructionTuningDataset(train_data)
    val_dataset = InstructionTuningDataset(val_data)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize accelerator
    accelerator = Accelerator()
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    unwrapped_model = accelerator.unwrap_model(model)

    device = accelerator.device

    best_val_loss = float("inf")
    epochs_no_improve = 0

    if accelerator.is_main_process:
        if log_wandb:
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model_name": model_type,
                    "scheme": scheme,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "model_save_dir": model_save_dir,
                    "train_dataset_path": train_dataset_path,
                    "peft": do_peft,
                },
            )

    eval_interval_step = max(1, len(train_dataloader) // 10)

    # Fine-tuning Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            outputs = unwrapped_model.forward_und(**batch)
            loss = outputs["loss"]

            # Backward pass and optimizer step
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                mean_loss = accelerator.gather_for_metrics(loss).mean().item()
                if accelerator.is_main_process:
                    if log_wandb:
                        wandb.log({"train_loss": mean_loss})
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {mean_loss:.4f}")

            if (step + 1) % eval_interval_step == 0:
                model.eval()
                total_val_loss = 0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_outputs = unwrapped_model.forward_und(**val_batch)
                        val_loss = val_outputs["loss"]
                        total_val_loss += val_loss

                mean_val_loss = torch.mean(accelerator.gather_for_metrics(total_val_loss)).item() / len(val_dataloader)
                if accelerator.is_main_process:
                    print(f"Step [{step+1}/{len(train_dataloader)}], Val Loss: {mean_val_loss:.4f}")
                    if log_wandb:
                        wandb.log({"val_loss": mean_val_loss})

                # Save the model
                accelerator.wait_for_everyone()
                save_dir = os.path.join(model_save_dir, f"epoch-{epoch+1}-step-{step+1}")

                if enable_zero3:
                    model.save_checkpoint(save_dir)
                elif accelerator.is_main_process:
                    accelerator.unwrap_model(model).save_pretrained(save_dir)

                model.train()  # Switch back to training mode

    if accelerator.is_main_process:
        if log_wandb:
            wandb.finish
    