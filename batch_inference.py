import json
import os
import pickle
import time
from io import BytesIO

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


class JanusInferenceDataset(Dataset):
    def __init__(self, df):
        self.images = df["image_bytes"]
        self.descriptions = df["long_desc"]
        self.subcategories = df["subcategory"]
        self.index = df.index
        self.sft_format = "<|User|>: <image_placeholder>\n\n{}\n\n<|Assistant|>:"
        self.max_str_len = 800
        self.max_detail_len = 50

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        image = Image.open(BytesIO(self.images[idx]))
        description = self.descriptions[idx]
        description = "; ".join([i.strip() for i in description.split(";") if len(i)<=self.max_detail_len])
        description = get_trimmed_prefix(description, self.max_str_len)

        instruction = """
        These are details of {} in the given image: {}
        Describe the garment in detail referring the information.
        """.format(
            self.subcategories[idx].lower(),
            description,
        ).strip()

        prompt = self.sft_format.format(instruction)
        index = self.index[idx]
        return {"prompt": prompt, "image": [image], "index": index}

def collate_fn(batch):
    # prompts, images, indices = zip(*batch)

    batch_dict = {key: [sample[key] for sample in batch] for key in batch[0]}
    
    prompts = batch_dict["prompt"]
    images = batch_dict["image"]
    indices = batch_dict["index"]
    indices = torch.tensor(indices)
    
    # Process batch using vl_chat_processor
    prepare_batch_inputs = vl_chat_processor.process_batch(prompt_batch=prompts, images_batch=images)
    
    return prepare_batch_inputs, indices

def get_trimmed_prefix(text, max_length):
    parts = [part.strip() for part in text.strip().split(';') if part.strip()]
    
    result = ""
    current = ""
    
    for part in parts:
        segment = part + "; "
        if len(current) + len(segment) > max_length:
            break
        current += segment

    return current.strip()

if __name__ == "__main__":
    train_dataset_path = "/data/charles/data/farfetch_full_250318.parquet"
    train_meta_path = "/data/charles/data/farfetch_meta_250314.json"
    result_path = "/data/charles/data/full_image_result_250321.json"

    with open(train_meta_path, "r") as file:
        train_meta = json.load(file)

    with open("/home/charles/VLM/Janus/notebook/category_map.pickle", "rb") as file:
        category_map = pickle.load(file)
    category_map.pop(())
    subcategory_set = set(i[-1] for i in category_map.keys())

    train_data = pd.read_parquet(train_dataset_path, engine="pyarrow", columns=None)
    train_data["long_desc"] = train_data["code"].apply(lambda x: train_meta[x]["long_desc"])
    train_data["image_index"] = train_data["image_path"].apply(lambda x: int(x.split("_")[-1].split(".")[0]))
    train_data["subcategory"] = train_data["code"].apply(lambda x: train_meta[x]["subcategory"])

    main_images = train_data[train_data["image_index"]==1]
    del train_data
    main_images = main_images[main_images["subcategory"].isin(subcategory_set)].reset_index(drop=True)

    # suffle or not
    main_images = main_images.sample(frac=1, random_state=42).reset_index(drop=True)

    # accelerate settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
    accelerator = Accelerator()
    device = accelerator.device

    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, cache_dir="/data/charles/huggingface/hub")
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, cache_dir="/data/charles/huggingface/hub"
    ).to(torch.bfloat16).to(device).eval()

    # create dataset
    batch_size = 10
    dataset = JanusInferenceDataset(main_images)
    del main_images
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=5, pin_memory=True)

    vl_gpt, dataloader = accelerator.prepare(vl_gpt, dataloader)
    unwrapped_model = accelerator.unwrap_model(vl_gpt)

    all_result = {}

    for step, (prepare_batch_inputs, indices) in enumerate(tqdm(dataloader, desc="Processing")):
        batch_inputs_embeds = unwrapped_model.prepare_inputs_embeds(**prepare_batch_inputs)

        with torch.no_grad():
            outputs = unwrapped_model.language_model.generate(
                inputs_embeds=batch_inputs_embeds,
                attention_mask=prepare_batch_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                # penalty_alpha=0.6, top_k=4
                do_sample=False,
                # use_cache=True,
            )

        result = {}
        outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)
        indices = accelerator.pad_across_processes(indices, dim=0, pad_index=-1)

        gathered_outputs = accelerator.gather(outputs)  # Gather all generated sequences
        gathered_indices = accelerator.gather(indices)

        if accelerator.is_main_process:
            merged_results = {}
            for output, idx in zip(gathered_outputs, gathered_indices):
                if idx.item() != -1:
                    answer = tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True)
                    merged_results[idx.item()] = answer.strip()

            all_result.update(merged_results)

            if step % 100 == 0:
                with open(result_path, "w") as file:
                    json.dump(all_result, file, indent=4, ensure_ascii=False)

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        with open(result_path, "w") as file:
            json.dump(all_result, file, indent=4, ensure_ascii=False)