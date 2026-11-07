import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from torch_geometric.nn.attention import QFormer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from qchem.model import GModel, GraphNet
from qchem.dataset import PGDS

from cosmosis.model import CModel


class LMMDataset(Dataset):
    def __init__(self, pyg_dataset, tokenizer, training=False):
        self.dataset = pyg_dataset
        self.tokenizer = tokenizer
        self.training = training
        
        self.prefix = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "The following is a molecular structure graph representation. Analyze its properties: "
        )
        self.suffix = (
            "\nPredict the solubility value of this molecule:<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def __len__(self):
        #return len(self.dataset)
        return 1000
        
    def __getitem__(self, i):
        data = self.dataset[i]
        if hasattr(data, 'y') and data.y is not None:
            val = data.y.item() if isinstance(data.y, torch.Tensor) else data.y
            target_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        else:
            target_str = ""

        prefix_tokens = self.tokenizer(self.prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer(self.suffix, add_special_tokens=False)
        
        if self.training and target_str:
            target_tokens = self.tokenizer(f"{target_str}<|eot_id|>", add_special_tokens=False)
            target_ids = target_tokens["input_ids"]
            target_mask = target_tokens["attention_mask"]
        else:
            target_ids = []
            target_mask = []

        return {
            "graph_data": data,
            "prefix_input_ids": torch.tensor(prefix_tokens["input_ids"], dtype=torch.long),
            "prefix_attention_mask": torch.tensor(prefix_tokens["attention_mask"], dtype=torch.long),
            "suffix_input_ids": torch.tensor(suffix_tokens["input_ids"], dtype=torch.long),
            "suffix_attention_mask": torch.tensor(suffix_tokens["attention_mask"], dtype=torch.long),
            "target_input_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_attention_mask": torch.tensor(target_mask, dtype=torch.long),
            "training": self.training
        }
        
class Llama(nn.Module):
    def __init__(self, model_id="akjindal53244/Llama-3.1-Storm-8B"):
        super().__init__()
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir='./model/huggingface'
        )

        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
        base_model.enable_input_require_grads()
        
        self.lora_config = LoraConfig(
            r=16,                                 
            lora_alpha=32,                          
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=0.05,                      
            bias="none",                             
            task_type=TaskType.CAUSAL_LM             
        )

        self.model = get_peft_model(base_model, self.lora_config)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable()
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


class LMM(nn.Module): 
    def __init__(self, llm, gnn, in_dim=256, 
                                 hidden_dim=512, 
                                 out_dim=256, 
                                 llm_dim=4096, 
                 output_dir='./data', device='cuda:0'):
        super().__init__()
        self.output_dir = output_dir
        self.device = device
        self.llm = llm
        self.gnn = gnn.to(device=self.device)
        
        self.gnn.layers.append(QFormer(input_dim=in_dim, 
                                       hidden_dim=hidden_dim, 
                                       output_dim=out_dim, 
                                       num_heads=8, 
                                       num_layers=2)).to(dtype=torch.bfloat16, device=self.device)
    
        self.projection_layer = nn.Linear(out_dim, llm_dim).to(dtype=torch.bfloat16, device=self.device)

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)
        
    def forward(self, graph_batch=None, prefix_input_ids=None, prefix_attention_mask=None, 
                suffix_input_ids=None, suffix_attention_mask=None, 
                target_input_ids=None, target_attention_mask=None, labels=None):
        
        embedding_layer = self.llm.get_input_embeddings()
        batch_size = prefix_input_ids.size(0)

        prefix_input_ids = prefix_input_ids.to(self.device)
        suffix_input_ids = suffix_input_ids.to(self.device)
        
        prefix_embed = embedding_layer(prefix_input_ids).to(dtype=torch.bfloat16)   
        suffix_embed = embedding_layer(suffix_input_ids).to(dtype=torch.bfloat16)   

        if target_input_ids is not None and target_input_ids.size(1) > 0:
            target_input_ids = target_input_ids.to(self.device)
            target_embed = embedding_layer(target_input_ids).to(dtype=torch.bfloat16)
            has_targets = True
        else:
            has_targets = False

        graph_batch = graph_batch.to(self.device)
        gnn_embed = self.gnn(graph_batch)  
        ptr = graph_batch.ptr

        all_embed = []
        all_mask = []
        all_labels = []
        
        for i in range(batch_size):
            single_graph_embed = gnn_embed[ptr[i]:ptr[i + 1]].to(dtype=torch.bfloat16)
            single_graph_proj = self.projection_layer(single_graph_embed)
            graph_mask = torch.ones(single_graph_proj.size(0), dtype=torch.long, device=self.device)
            
            if has_targets:
                t_ids = target_input_ids[i].to(device=self.device, dtype=torch.long)
                t_mask = target_attention_mask[i].to(device=self.device, dtype=torch.long)
                t_emb = target_embed[i]
            else:
                t_ids = torch.tensor([], dtype=torch.long, device=self.device)
                t_mask = torch.tensor([], dtype=torch.long, device=self.device)
                t_emb = torch.tensor([], dtype=torch.bfloat16, device=self.device).reshape(0, prefix_embed.size(-1))

            p_mask = prefix_attention_mask[i].to(device=self.device, dtype=torch.long)
            s_mask = suffix_attention_mask[i].to(device=self.device, dtype=torch.long)

            embed_segments = [prefix_embed[i], single_graph_proj, suffix_embed[i]]
            if has_targets:
                embed_segments.append(t_emb)

            mask_segments = [p_mask, graph_mask, s_mask]
            if has_targets:
                mask_segments.append(t_mask)

            prompt_len = prefix_input_ids[i].size(0) + single_graph_proj.size(0) + suffix_input_ids[i].size(0)
            prompt_label_mask = torch.full((prompt_len,), -100, dtype=torch.long, device=self.device)

            label_segments = [prompt_label_mask]
            if has_targets:
                label_segments.append(t_ids)
            
            merged_embed = torch.cat(embed_segments, dim=0)
            merged_mask = torch.cat(mask_segments, dim=0)
            merged_label = torch.cat(label_segments, dim=0)
            
            all_embed.append(merged_embed)
            all_mask.append(merged_mask)
            all_labels.append(merged_label)

        input_embed = pad_sequence(all_embed, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(all_mask, batch_first=True, padding_value=0)
        final_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100) if has_targets else None

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return self.llm.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=final_labels,
            return_dict=True
        )


    @staticmethod
    def lmm_collate_fn(batch):
        return {
            "graph_batch": Batch.from_data_list([x["graph_data"] for x in batch]),
            "prefix_input_ids": pad_sequence([x["prefix_input_ids"] for x in batch], batch_first=True, padding_value=0),
            "prefix_attention_mask": pad_sequence([x["prefix_attention_mask"] for x in batch], batch_first=True, padding_value=0),
            "suffix_input_ids": pad_sequence([x["suffix_input_ids"] for x in batch], batch_first=True, padding_value=0),
            "suffix_attention_mask": pad_sequence([x["suffix_attention_mask"] for x in batch], batch_first=True, padding_value=0),
            "target_input_ids": pad_sequence([x["target_input_ids"] for x in batch], batch_first=True, padding_value=0),
            "target_attention_mask": pad_sequence([x["target_attention_mask"] for x in batch], batch_first=True, padding_value=0),
            "labels": torch.ones(len(batch), dtype=torch.long)} # tells Trainer labels are present
            
    def train_model(self, llm_ds):

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4, 
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            num_train_epochs=1,
            bf16=True, 
            fp16=False,
            max_grad_norm=0.3,
            report_to="none",
            remove_unused_columns=False)
    
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=llm_ds,
            data_collator=self.lmm_collate_fn)
        
        trainer.train()

    @torch.no_grad()
    def infer(self, prompts, batch_size=4, max_new_tokens=30):
        """
        Accepts:
            - A single dictionary from LMMDataset (e.g., lmm_ds[0])
            - A list of dictionaries from LMMDataset
            - The full LMMDataset object itself
        Runs inference in parallel using automated batch collation.
        """
        self.eval()

        if isinstance(prompts, LMMDataset):
            items_list = [prompts[i] for i in range(len(prompts))]
        elif isinstance(prompts, dict):
            items_list = [prompts]
        elif isinstance(prompts, list) or (hasattr(prompts, "__iter__") and not isinstance(prompts, dict)):
            items_list = list(prompts)
        else:
            raise TypeError("Expected input to be a dictionary, a list of dictionaries, or an LMMDataset instance.")

        if not items_list:
            return []

        eval_loader = DataLoader(items_list, batch_size=batch_size, shuffle=False, collate_fn=self.lmm_collate_fn)
        all_decoded_outputs = []

        for batch in eval_loader:
            p_ids = batch["prefix_input_ids"].to(dtype=torch.long, device=self.device)
            p_mask = batch["prefix_attention_mask"].to(dtype=torch.long, device=self.device)
            s_ids = batch["suffix_input_ids"].to(dtype=torch.long, device=self.device)
            s_mask = batch["suffix_attention_mask"].to(dtype=torch.long, device=self.device)

            graph_batch = batch["graph_batch"].to(device=self.device)
            graph_batch.x = graph_batch.x.to(dtype=torch.bfloat16)
            if hasattr(graph_batch, 'edge_attr') and graph_batch.edge_attr is not None:
                graph_batch.edge_attr = graph_batch.edge_attr.to(dtype=torch.bfloat16)
            if hasattr(graph_batch, 'pos') and graph_batch.pos is not None:
                graph_batch.pos = graph_batch.pos.to(dtype=torch.bfloat16)

            embedding_layer = self.llm.get_input_embeddings()
            prefix_embed = embedding_layer(p_ids)   # [B, seq_len_p, llm_dim]
            suffix_embed = embedding_layer(s_ids)   # [B, seq_len_s, llm_dim]
            
            gnn_embed = self.gnn(graph_batch)       # [Total_Nodes_In_Batch, out_channels]
            
            ptr = graph_batch.ptr
            current_batch_size = p_ids.size(0)
            
            all_graph_proj = []
            all_graph_mask = []
            for i in range(current_batch_size):
                single_graph_embed = gnn_embed[ptr[i]:ptr[i + 1]]
                single_graph_proj = self.projection_layer(single_graph_embed)
                
                if torch.isnan(single_graph_proj).any() or torch.isinf(single_graph_proj).any():
                    single_graph_proj = torch.where(torch.isnan(single_graph_proj) | torch.isinf(single_graph_proj), torch.zeros_like(single_graph_proj), single_graph_proj)
                
                graph_mask_item = torch.ones(single_graph_proj.size(0), dtype=torch.long, device=self.device)
                
                all_graph_proj.append(single_graph_proj)
                all_graph_mask.append(graph_mask_item)

            graph_proj_padded = pad_sequence(all_graph_proj, batch_first=True, padding_value=0.0)
            graph_mask_padded = pad_sequence(all_graph_mask, batch_first=True, padding_value=0)

            input_embed = torch.cat([prefix_embed, graph_proj_padded, suffix_embed], dim=1)
            attention_mask = torch.cat([p_mask, graph_mask_padded, s_mask], dim=1)

            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = self.llm.model.generate(
                inputs_embeds=input_embed,
                attention_mask=attention_mask,
                position_ids=position_ids, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=[
                    self.llm.tokenizer.eos_token_id,
                    self.llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
            
            for sequence in outputs:
                decoded = self.llm.tokenizer.decode(sequence, skip_special_tokens=True)
                all_decoded_outputs.append(decoded)

        if isinstance(prompts, dict):
            return all_decoded_outputs[0]
            
        return all_decoded_outputs


