import math

import torch
import torch.nn as nn

# --- 核心元件 ---

class MultiHeadAttention(nn.Module):
    """
    多頭注意力機制實現
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Args:
            d_model (int): 模型的總維度 (必須能被 nhead 整除)
            nhead (int): 注意力頭的數量
            dropout (float): Dropout 的機率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每個頭的維度

        # 定義 Q, K, V 和輸出的線性層
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (Tensor): shape (batch_size, seq_len_q, d_model)
            key (Tensor): shape (batch_size, seq_len_k, d_model)
            value (Tensor): shape (batch_size, seq_len_v, d_model)
            mask (Tensor, optional): 遮罩，用於防止注意力關注到不該關注的位置. Defaults to None.
                                     Shape: (batch_size, 1, 1, seq_len_k) for padding mask or
                                            (batch_size, 1, seq_len_q, seq_len_k) for subsequent mask
        """
        batch_size = query.size(0)

        # 1. 通過線性層並重塑為多頭形式
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, nhead, d_k) -> (batch_size, nhead, seq_len, d_k)
        q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 2. 計算注意力分數 (Scaled Dot-Product Attention)
        # scores shape: (batch_size, nhead, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 應用遮罩 (Masking)
        if mask is not None:
            # 使用一個非常小的數來填充被遮罩的位置，這樣 softmax 後會趨近於 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. 對分數進行 softmax 得到注意力權重
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 將權重應用於 V
        # context shape: (batch_size, nhead, seq_len_q, d_k)
        context = torch.matmul(attn_weights, v)

        # 6. 將多頭的結果串接起來並通過最後的線性層
        # context shape: (batch_size, seq_len_q, nhead, d_k) -> (batch_size, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    位置全域性前饋網路
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    """
    位置編碼
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1) # shape: (max_len, 1, d_model)
        
        # 將 pe 註冊為 buffer，這樣它不會被視為模型參數，但會隨模型移動 (e.g., to(device))
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (seq_len, batch_size, d_model)
        """
        # 將位置編碼加到輸入的嵌入向量上
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 組合元件：Encoder & Decoder Layer ---

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. 多頭自注意力層 + Add & Norm
        # Q, K, V 都來自同一個來源 src
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # 2. 前饋網路層 + Add & Norm
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 第一個注意力層：遮罩後的自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        # 第二個注意力層：與編碼器輸出的交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. 遮罩後的自注意力層 + Add & Norm
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # 2. 交叉注意力層 (Encoder-Decoder Attention) + Add & Norm
        # Q 來自解碼器自身，K 和 V 來自編碼器的輸出 (memory)
        cross_attn_output, _ = self.cross_attn(query=tgt, key=memory, value=memory, mask=memory_mask)
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)

        # 3. 前饋網路層 + Add & Norm
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)

        return tgt

# --- 最終模型：Transformer ---

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ff, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 嵌入層和位置編碼
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # 編碼器
        encoder_layer = EncoderLayer(d_model, nhead, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 解碼器
        decoder_layer = DecoderLayer(d_model, nhead, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 最終的線性輸出層
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        # 初始化權重
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        # 生成一個上三角矩陣，用於遮蔽未來的 token
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src (Tensor): aource sequence, shape (src_seq_len, batch_size)
            tgt (Tensor): target sequence, shape (tgt_seq_len, batch_size)
            src_padding_mask (Tensor): source padding mask, shape (batch_size, src_seq_len)
            tgt_padding_mask (Tensor): target padding mask, shape (batch_size, tgt_seq_len)
            memory_key_padding_mask (Tensor): aource padding mask for cross-attention, shape (batch_size, src_seq_len)
        """
        
        # 1. 嵌入和位置編碼
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 2. 生成解碼器自注意力所需的遮罩
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        # 3. Encoder 處理
        # PyTorch 內建的 TransformerEncoder 和 Decoder 要求 (seq_len, batch_size, d_model)
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # 4. Decoder 處理
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        # 5. 輸出層
        output = self.fc_out(output)
        
        return output

# --- 使用範例 ---

if __name__ == '__main__':
    # 參數設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SRC_VOCAB_SIZE = 5000
    TGT_VOCAB_SIZE = 5000
    D_MODEL = 512       # 模型維度
    NHEAD = 8           # 注意力頭數
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    D_FF = 2048         # 前饋網路中間層維度
    MAX_LEN = 100
    DROPOUT = 0.1
    
    # 建立模型
    transformer = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    # 建立假的輸入資料
    # 注意：這裡的輸入格式是 (seq_len, batch_size)
    src_seq_len = 20
    tgt_seq_len = 18
    batch_size = 4
    
    src = torch.randint(1, SRC_VOCAB_SIZE, (src_seq_len, batch_size)).to(device)
    tgt = torch.randint(1, TGT_VOCAB_SIZE, (tgt_seq_len, batch_size)).to(device)

    # 執行前向傳播
    transformer.train() # 設為訓練模式
    output = transformer(src, tgt)
    
    print("Transformer 模型已成功建立並執行！")
    print("輸入 src shape:", src.shape)
    print("輸入 tgt shape:", tgt.shape)
    print("輸出 output shape:", output.shape) # 應為 (tgt_seq_len, batch_size, TGT_VOCAB_SIZE)
    
    # 檢查參數數量
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"模型總參數數量: {num_params:,}")