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
from cosmosis.model import CModel, FFNet
from qchem.dataset import PGDS


class LMMDataset(Dataset):
    def __init__(self, pyg_dataset, tokenizer, device='cuda:0', training=False):
        self.device = device
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
        return 100

    def __getitem__(self, i):
        data = self.dataset[i]
        target_val = hasattr(data, 'y')

        prefix_tokens = self.tokenizer(self.prefix, add_special_tokens=False, return_tensors="pt")
        
        suffix_text = self.suffix
        if self.training:
            suffix_text += f"{target_val}<|eot_id|>"
        suffix_tokens = self.tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt")

        return {
            "graph_data": data.to(self.device),
            "prefix_input_ids": prefix_tokens.input_ids.squeeze(0).to(self.device),
            "prefix_attention_mask": prefix_tokens.attention_mask.squeeze(0).to(self.device),
            "suffix_input_ids": suffix_tokens.input_ids.squeeze(0).to(self.device),
            "suffix_attention_mask": suffix_tokens.attention_mask.squeeze(0).to(self.device),
            "training": self.training
        }


class Llama(nn.Module):
    def __init__(self):
        super().__init__()
        model_id = "akjindal53244/Llama-3.1-Storm-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
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
        
        self.lora_config = LoraConfig(
            r=16,                                 
            lora_alpha=32,                          
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=0.05,                      
            bias="none",                             
            task_type=TaskType.CAUSAL_LM             
        )

        self.model = get_peft_model(base_model, self.lora_config)
        self.model.gradient_checkpointing_enable()
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


class LMM(nn.Module): 
    def __init__(self, llm, gnn, in_dim=256, 
                 hidden_dim=512, out_dim=256, llm_dim=4096, output_dir='./data', device='cuda:0'):
        super().__init__()
        self.output_dir = output_dir
        self.device = device
        self.llm = llm.to(self.device)
        self.gnn = gnn.to(self.device)
        
        self.gnn.layers.append(QFormer(input_dim=in_dim, 
                                       hidden_dim=hidden_dim, 
                                       output_dim=out_dim, 
                                       num_heads=8, 
                                       num_layers=2))
        self.gnn.activation = None
        self.gnn.pooling = None
        self.gnn.ffn = None
    
        
        self.projection_layer = FFNet({'model_name': 'simple', 
                                       'in_channels': out_dim, 
                                       'out_channels': llm_dim,
                                       'device': self.device})

    def forward(self, graph_batch=None, prefix_input_ids=None, prefix_attention_mask=None, 
                suffix_input_ids=None, suffix_attention_mask=None, labels=None):
        
        embedding_layer = self.llm.get_input_embeddings().to(self.device)
        batch_size = prefix_input_ids.size(0)

        prefix_embed = embedding_layer(prefix_input_ids)   # [B, Seq_Len_P, LLM_EMB]
        suffix_embed = embedding_layer(suffix_input_ids)   # [B, Seq_Len_S, LLM_EMB]

        # [Total_Nodes_In_Batch, OUT_CHANNELS]
        gnn_embed = self.gnn(graph_batch)  
        ptr = graph_batch.ptr

        all_embed = []
        all_mask = []
        for i in range(batch_size):
            single_graph_embed = gnn_embed[ptr[i]:ptr[i + 1]]
            single_graph_proj = self.projection_layer(single_graph_embed)
            graph_mask = torch.ones(single_graph_proj.size(0), dtype=torch.long, device=self.device)
            
            merged_embed = torch.cat([
                prefix_embed[i], 
                single_graph_proj, 
                suffix_embed[i]
            ], dim=0)
            
            merged_mask = torch.cat([
                prefix_attention_mask[i],
                graph_mask, 
                suffix_attention_mask[i]
            ], dim=0)
            
            all_embed.append(merged_embed)
            all_mask.append(merged_mask)

        input_embed = pad_sequence(all_embed, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(all_mask, batch_first=True, padding_value=0)

        if labels is not None:
            final_labels = []
            for i in range(batch_size):
                graph_len = ptr[i + 1] - ptr[i]
                ignore_len = prefix_input_ids[i].size(0) + graph_len
                suffix_len = suffix_input_ids[i].size(0)
                
                label_seq = torch.full((ignore_len + suffix_len,), -100, dtype=torch.long, device=self.device)
                label_seq[-suffix_len:] = suffix_input_ids[i]
                final_labels.append(label_seq)
                
            labels = pad_sequence(final_labels, batch_first=True, padding_value=-100)

        return self.llm.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def train_model(self, pyg_dataset):
        llm_ds = LMMDataset(
            pyg_dataset=pyg_dataset,
            tokenizer=self.llm.tokenizer,
            training=True
        )

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
            warmup_ratio=0.03,
            report_to="none",
            remove_unused_columns=False
        )
        
        def custom_collate(batch):
            return {
                "graph_batch": Batch.from_data_list([x["graph_data"] for x in batch]),
                "prefix_input_ids": pad_sequence([x["prefix_input_ids"] for x in batch], batch_first=True, padding_value=0),
                "prefix_attention_mask": pad_sequence([x["prefix_attention_mask"] for x in batch], batch_first=True, padding_value=0),
                "suffix_input_ids": pad_sequence([x["suffix_input_ids"] for x in batch], batch_first=True, padding_value=0),
                "suffix_attention_mask": pad_sequence([x["suffix_attention_mask"] for x in batch], batch_first=True, padding_value=0),
                "labels": torch.ones(len(batch)) 
            }

        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=llm_ds,
            data_collator=custom_collate
        )
        trainer.train()

    def infer(self, prompt):
        self.eval()
        self.gnn.eval()
        self.projection_layer.eval()
        eval_ds = LMMDataset(
            pyg_dataset=[prompt], # single molecule
            tokenizer=self.llm.tokenizer,
            training=False
        )
    
        graph = prompt["graph_data"].to(self.device) # single Data obj
        p_ids = prompt["prefix_input_ids"].to(self.device)
        p_mask = prompt["prefix_attention_mask"].to(self.device)
        s_ids = prompt["suffix_input_ids"].to(self.device)
        s_mask = prompt["suffix_attention_mask"].to(self.device)

        with torch.no_grad():
            embedding_layer = self.llm.get_input_embeddings().to(self.device)
            prefix_embed = embedding_layer(p_ids)
            suffix_embed = embedding_layer(s_ids)
            loader = DataLoader([graph], batch_size=1)
            graph = next(iter(loader))
            graph_embed = self.gnn(graph)
            graph_proj = self.projection_layer(graph_embed)

            graph_mask = torch.ones(graph_proj.size(0), dtype=torch.long, device=self.device)

            input_embed = torch.cat([prefix_embed, graph_proj, suffix_embed], dim=0).unsqueeze(0)
            attention_mask = torch.cat([p_mask, graph_mask, s_mask], dim=0).unsqueeze(0)

            outputs = self.llm.model.generate(
                inputs_embeds=input_embed,
                attention_mask=attention_mask,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=[
                    self.llm.tokenizer.eos_token_id,
                    self.llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
            
        return outputs

