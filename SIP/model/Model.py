import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class InitiativeRecognizer(nn.Module):
    """
    Implementation of the initiative recognizer (Section 3.2.2 of IDPT paper).
    - Dialogue Encoder: BERT encoding of concatenated input S = [K; history]
    - Initiative Recognizer: Multi-head attention over learned prefix queries
    - Output: P_j, softmax distribution over initiative labels
    """
    def __init__(self,
                 bert_model_name: str = 'bert-base-uncased',
                 num_initiatives: int = 4,
                 prefix_length: int = 5,
                 decoder_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12):
        super().__init__()
        # 1. Dialogue Encoder (Eq.1)
        self.encoder = BertModel.from_pretrained(bert_model_name)
        # 2. Initiative-aware prefixes: one per initiative label
        #    Each prefix is a learnable tensor of shape [num_initiatives, prefix_length, hidden_size]
        self.prefixes = nn.Parameter(
            torch.randn(num_initiatives, prefix_length, hidden_size)
        )
        # 3. Multi-head attention parameters (shared keys/values from encoder outputs)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                               num_heads=num_attention_heads,
                                               batch_first=True)
        # 4. Fusion MLP (Eq.7 & Eq.8)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * prefix_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        # 5. Max-pooling and classifier (Eq.9-11)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_initiatives)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Encode dialogue history + knowledge K
        # H: [batch_size, seq_len, hidden_size]
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, S, H)

        batch_size = H.size(0)
        P_logits = []
        # For each initiative label j, compute attention and fusion
        for j in range(self.prefixes.size(0)):
            # Prefix query: shape [prefix_length, hidden_size]
            prefix_j = self.prefixes[j]  # (P, H)
            # Expand to batch
            Q = prefix_j.unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, H)
            # Key, Value are H
            # Multi-head attention: query=Q, key=H, value=H
            # attn_output: (B, P, H)
            attn_output, _ = self.attention(query=Q,
                                           key=H,
                                           value=H,
                                           key_padding_mask=~attention_mask.bool())
            # Flatten P x H and fuse
            z_flat = attn_output.view(batch_size, -1)  # (B, P*H)
            h_prime = self.fusion(z_flat)  # (B, H)
            h_prime = self.layer_norm(h_prime + prefix_j.mean(dim=0))  # Eq.8
            # Collect for max-pooling
            P_logits.append(h_prime)

        # Stack and max-pool over initiatives dimension
        # H_it: (B, num_initiatives, H)
        H_it = torch.stack(P_logits, dim=1)
        # Max-pooling over second dim -> (B, H)
        H_it_pooled, _ = H_it.max(dim=1)
        # Classifier -> logits over initiatives
        logits = self.classifier(H_it_pooled)  # (B, num_initiatives)
        probs = nn.functional.softmax(logits, dim=-1)  # P_j (Eq.11)
        return probs, logits

# Example usage:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# texts = ['[K] ... [SEP] user: ... system: ...']
# enc = tokenizer(texts, return_tensors='pt', padding=True)
# model = InitiativeRecognizer()
# P_j, logits = model(enc['input_ids'], enc['attention_mask'])
# Loss: nn.CrossEntropyLoss()(logits, true_labels)

