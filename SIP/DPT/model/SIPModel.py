import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SIPRecognizer(nn.Module):
    """
    SIP (System Initiative Prediction) Recognizer for binary classification.
    - Dialogue Encoder: BERT encoding of concatenated input S = [K; history]
    - Initiative Recognizer: Multi-head attention over learned prefix queries
    - Output: Binary classification (SIP=0 or SIP=1)
    """
    def __init__(self,
                 bert_model_name: str = 'bert-base-uncased',
                 prefix_length: int = 5,
                 num_attention_heads: int = 12,
                 freeze_encoder: bool = True,
                 num_initiatives: int = 2):
        super().__init__()
        self.num_initiatives = num_initiatives
        
        # 1. Dialogue Encoder (Eq.1)
        self.encoder = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # 2. Initiative-aware prefixes: one per SIP label (0 and 1)
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
            nn.Linear(hidden_size, num_initiatives)  # num_initiatives for binary classification
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Encode dialogue history + knowledge K
        # H: [batch_size, seq_len, hidden_size]
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, S, H)

        batch_size = H.size(0)
        P_logits = []
        # For each SIP label j (0 and 1), compute attention and fusion
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
            z_flat = attn_output.reshape(batch_size, -1)  # (B, P*H)
            h_prime = self.fusion(z_flat)  # (B, H)
            h_prime = self.layer_norm(h_prime + prefix_j.mean(dim=0))  # Eq.8
            # Collect for max-pooling
            P_logits.append(h_prime)

        # Stack and max-pool over SIP dimension
        # H_it: (B, num_initiatives, H)
        H_it = torch.stack(P_logits, dim=1)
        # Max-pooling over second dim -> (B, H)
        H_it_pooled, _ = H_it.max(dim=1)
        # Classifier -> logits over SIP labels
        logits = self.classifier(H_it_pooled)  # (B, num_initiatives)
        probs = nn.functional.softmax(logits, dim=-1)  # P_j (Eq.11)
        return probs, logits

# スモークテスト用の関数
def smoke_test():
    """
    モデルの形状チェックと推論テスト
    """
    from SIP.model.util import create_class_weights, get_loss
    
    print("=== SIP Model Smoke Test ===")
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデル初期化
    num_initiatives = 2
    model = SIPRecognizer(num_initiatives=num_initiatives)
    model.to(device)
    model.eval()  # 推論モード
    
    # サンプルデータ作成
    batch_size = 4
    seq_length = 128
    
    # ダミーの入力データ
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    # 推論実行
    with torch.no_grad():
        probs, logits = model(input_ids, attention_mask)
    
    print(f"Output shapes:")
    print(f"  logits: {logits.shape}")  # 期待値: (B, num_initiatives)
    print(f"  probs: {probs.shape}")    # 期待値: (B, num_initiatives)
    
    # 形状チェック
    assert logits.shape == (batch_size, num_initiatives), f"Expected logits shape (B, {num_initiatives}), got {logits.shape}"
    assert probs.shape == (batch_size, num_initiatives), f"Expected probs shape (B, {num_initiatives}), got {probs.shape}"
    
    # 確率の合計が1になることを確認
    prob_sums = probs.sum(dim=1)
    print(f"Probability sums: {prob_sums}")
    assert torch.allclose(prob_sums, torch.ones(batch_size).to(device)), "Probabilities should sum to 1"
    
    # クラス重みの例（動的クラス数に対応）
    class_counts = [2309, 357]  # 2クラスの場合
    class_weights = create_class_weights(class_counts, method='balanced')
    class_weights = class_weights.to(device)
    
    print(f"\nClass weights (balanced):")
    for i, weight in enumerate(class_weights):
        print(f"  w{i}: {weight:.3f}")
    
    # ダミーのターゲット
    targets = torch.randint(0, num_initiatives, (batch_size,)).to(device)
    
    # 損失計算の例（utilからインポート）
    loss = get_loss(logits, targets, class_weights)
    print(f"\nLoss with class weights: {loss.item():.4f}")
    
    # 重みなし損失との比較
    loss_no_weights = get_loss(logits, targets, None)
    print(f"Loss without weights: {loss_no_weights.item():.4f}")
    
    print("\n✅ Smoke test passed!")

# 使用例
if __name__ == "__main__":
    # スモークテスト実行
    smoke_test()
    
    # 学習時の使用例（コメントアウト）
    """
    from SIP.model.util import create_class_weights, get_loss
    
    # データ準備
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = ['[Knowledge] ... [SEP] User: ... System: ... [SEP] User: ...']
    enc = tokenizer(texts, return_tensors='pt', padding=True)
    
    # モデル初期化
    num_initiatives = 2
    model = SIPRecognizer(num_initiatives=num_initiatives)
    
    # クラス重み計算（動的クラス数に対応）
    class_counts = [2309, 357]  # 各クラスのサンプル数
    class_weights = create_class_weights(class_counts, method='balanced')
    class_weights = class_weights.to(device)  # デバイスに移動
    
    # 推論
    probs, logits = model(enc['input_ids'], enc['attention_mask'])
    
    # 損失計算（学習時）
    targets = torch.tensor([0, 1, 0, 1])  # ダミーのターゲット
    loss = get_loss(logits, targets, class_weights)
    """ 