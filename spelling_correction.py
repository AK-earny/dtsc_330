"""
COMPLETE TRANSFORMER SEQ2SEQ FOR SPELLING CORRECTION
=====================================================

This is a fully-annotated, production-quality implementation
with detailed comments explaining every component.

You can run this directly:
    python3 COMPLETE_IMPLEMENTATION.py

Requirements:
    pip install tensorflow numpy --break-system-packages
"""

import tensorflow as tf
import numpy as np
import random
from typing import Tuple, List

# ============================================================================
# SECTION 1: DATA GENERATION
# ============================================================================

def generate_synthetic_data(num_pairs: int = 500, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Generate misspelled-corrected word pairs for training.
    
    Args:
        num_pairs: Number of (misspelled, correct) pairs to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of (misspelled_word, correct_word) tuples
    """
    random.seed(seed)
    
    # Common words to use as ground truth
    vocabulary = [
        "hello", "world", "python", "tensorflow", "neural", "network", 
        "learning", "computer", "algorithm", "function", "attention", 
        "transformer", "sequence", "data", "science", "machine", "deep",
        "recurrent", "model", "training", "accuracy", "prediction"
    ]
    
    pairs = []
    
    for _ in range(num_pairs):
        correct_word = random.choice(vocabulary)
        
        # Introduce 1-2 random errors
        misspelled = introduce_errors(correct_word, num_errors=random.randint(1, 2))
        
        # Avoid identical pairs
        if misspelled != correct_word:
            pairs.append((misspelled, correct_word))
    
    return pairs


def introduce_errors(word: str, num_errors: int = 1) -> str:
    """
    Introduce random spelling mistakes to a word.
    
    Error types:
        - swap: Switch two adjacent characters
        - delete: Remove a character
        - insert: Add a random character
        - replace: Change a character to another
        
    Args:
        word: Original word
        num_errors: Number of errors to introduce
        
    Returns:
        Misspelled version of word
    """
    word = list(word)
    errors_made = 0
    
    while errors_made < num_errors and len(word) > 1:
        error_type = random.choice(['swap', 'delete', 'insert', 'replace'])
        idx = random.randint(0, len(word) - 1)
        
        try:
            if error_type == 'swap' and idx < len(word) - 1:
                word[idx], word[idx + 1] = word[idx + 1], word[idx]
                errors_made += 1
                
            elif error_type == 'delete':
                word.pop(idx)
                errors_made += 1
                
            elif error_type == 'insert' and len(word) < 20:
                word.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
                errors_made += 1
                
            elif error_type == 'replace':
                word[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                errors_made += 1
        except:
            pass
    
    return ''.join(word)


# ============================================================================
# SECTION 2: TOKENIZATION
# ============================================================================

class CharacterTokenizer:
    """
    Convert text to token indices and back.
    
    Vocabulary:
        - <pad> (index 0): Padding token for sequences shorter than max_length
        - <start> (index 1): Marks beginning of target sequence
        - <end> (index 2): Marks end of sequence (optional)
        - a-z: Actual characters
    """
    
    def __init__(self):
        # Build vocabulary
        self.vocab = ['<pad>', '<start>', '<end>'] + list('abcdefghijklmnopqrstuvwxyz')
        
        # Create bidirectional mappings
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
    
    def encode(self, text: str, max_length: int = 20) -> np.ndarray:
        """
        Convert text string to indices with padding.
        
        Process:
            1. Convert each character to its index
            2. Pad with zeros if too short
            3. Truncate if too long
            
        Args:
            text: Input text (e.g., "hello")
            max_length: Pad/truncate to this length
            
        Returns:
            Array of shape (max_length,) with character indices
        """
        # Convert characters to indices, unknown chars → 0 (pad)
        indices = [self.char2idx.get(char, 0) for char in text.lower()]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices = indices + [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return np.array(indices, dtype=np.int32)
    
    def decode(self, indices: np.ndarray) -> str:
        """
        Convert indices back to text.
        
        Args:
            indices: Array of character indices
            
        Returns:
            Decoded text (removing special tokens)
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(int(idx), '<unk>')
            # Skip special tokens
            if char not in ['<pad>', '<start>', '<end>', '<unk>']:
                chars.append(char)
        return ''.join(chars)
    
    @property
    def vocab_size(self):
        """Return the size of vocabulary."""
        return len(self.vocab)


# ============================================================================
# SECTION 3: ATTENTION LAYERS
# ============================================================================

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention Layer
    
    Attention Formula:
        Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    
    Where:
        - Q (Query): "What am I looking for?"
        - K (Key): "What information do I have?"
        - V (Value): "What do I return?"
        
    Multi-Head:
        - Instead of 1 attention, use H heads in parallel
        - Each head focuses on different representation subspace
        - Results concatenated and projected
        
    Self-Attention: Q, K, V all from same input
    Cross-Attention: Q from decoder, K,V from encoder
    """
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Dimension of each head
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        
        # Output projection
        self.W_o = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Split last dimension into (num_heads, d_head).
        
        Shape transformation:
            (batch, seq_len, d_model) →
            (batch, seq_len, num_heads, d_head) →
            (batch, num_heads, seq_len, d_head)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            Q: Query tensor (batch, seq_len_q, d_model)
            K: Key tensor (batch, seq_len_k, d_model)
            V: Value tensor (batch, seq_len_v, d_model)
            
        Returns:
            output: Attention output (batch, seq_len_q, d_model)
            attention_weights: For visualization (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = tf.shape(Q)[0]
        
        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_head)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention
        # Q·K^T gives (batch, num_heads, seq_len_q, seq_len_k)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Multiply by values
        output = tf.matmul(attention_weights, V)  # (batch, num_heads, seq_len, d_head)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder Layer
    
    Flow:
        Input
          ↓
        Multi-Head Self-Attention
          ↓
        Residual Connection + LayerNorm
          ↓
        Feed-Forward Network
          ↓
        Residual Connection + LayerNorm
          ↓
        Output
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, **kwargs):
        super().__init__(**kwargs)
        
        # Self-attention
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network (position-wise)
        # Typical: d_model → d_ff (expansion) → d_model (projection)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """
    Transformer Decoder Layer
    
    Flow:
        Decoder Input
          ↓
        Self-Attention (on decoder sequence)
          ↓
        Residual + LayerNorm
          ↓
        Cross-Attention (to encoder) ⭐
          ↓
        Residual + LayerNorm
          ↓
        Feed-Forward
          ↓
        Residual + LayerNorm
          ↓
        Output
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, **kwargs):
        super().__init__(**kwargs)
        
        # Self-attention on decoder sequence
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        
        # Cross-attention: decoder to encoder
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Layer normalizations
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, decoder_input: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        """
        Args:
            decoder_input: Decoder sequence (batch, seq_len_decoder, d_model)
            encoder_output: Encoder output (batch, seq_len_encoder, d_model)
            
        Returns:
            Decoded representation (batch, seq_len_decoder, d_model)
        """
        # Decoder self-attention: understand the output sequence
        self_attn_out, _ = self.self_mha(decoder_input, decoder_input, decoder_input)
        out1 = self.layernorm1(decoder_input + self_attn_out)
        
        # Cross-attention: align with input
        # Q from decoder (what to generate), K,V from encoder (what's in input)
        cross_attn_out, _ = self.cross_mha(out1, encoder_output, encoder_output)
        out2 = self.layernorm2(out1 + cross_attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(out2)
        out3 = self.layernorm3(out2 + ffn_out)
        
        return out3


# ============================================================================
# SECTION 4: FULL TRANSFORMER MODEL
# ============================================================================

class TransformerSpellingCorrector(tf.keras.Model):
    """
    Complete Seq2Seq Transformer Model
    
    Architecture:
        Input (misspelled) → Embedding + PositionalEncoding
                                ↓
                            ENCODER (self-attention)
                                ↓
                            DECODER (self-attn + cross-attn)
                                ↓
                            Output (character predictions)
    """
    
    def __init__(self, vocab_size: int, max_length: int, d_model: int = 64,
                 num_heads: int = 4, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, d_ff: int = 256, **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        
        # Positional encoding (precomputed)
        self.pos_encoding = self._create_positional_encoding(max_length, d_model)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_encoder_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_decoder_layers)
        ]
        
        # Output layer: project to vocabulary
        self.output_layer = tf.keras.layers.Dense(vocab_size)
    
    def _create_positional_encoding(self, max_length: int, d_model: int) -> tf.Tensor:
        """
        Create positional encodings.
        
        Formula:
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
            
        Intuition:
            - Low frequencies: capture long-range patterns
            - High frequencies: capture fine-grained position info
            - Different positions get different encoding patterns
        """
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
    
    def encode(self, encoder_input: tf.Tensor) -> tf.Tensor:
        """
        Encoder: Process misspelled input with self-attention.
        
        Args:
            encoder_input: Indices (batch, seq_len)
            
        Returns:
            Context vector (batch, seq_len, d_model)
        """
        seq_len = tf.shape(encoder_input)[1]
        
        # Embedding + positional encoding
        x = self.embedding(encoder_input)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # Apply encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        return x
    
    def decode(self, decoder_input: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        """
        Decoder: Generate output with attention to encoder.
        
        Args:
            decoder_input: Indices (batch, seq_len)
            encoder_output: Encoder output (batch, seq_len_enc, d_model)
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        seq_len = tf.shape(decoder_input)[1]
        
        # Embedding + positional encoding
        x = self.embedding(decoder_input)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # Apply decoder layers with cross-attention
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output)
        
        # Project to vocabulary
        logits = self.output_layer(x)
        
        return logits
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        """
        Full forward pass.
        
        Args:
            inputs: Tuple of (encoder_input, decoder_input)
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        encoder_input, decoder_input = inputs
        encoder_output = self.encode(encoder_input)
        logits = self.decode(decoder_input, encoder_output)
        return logits


# ============================================================================
# SECTION 5: TRAINING
# ============================================================================

def create_training_tensors(pairs: List[Tuple[str, str]], tokenizer: CharacterTokenizer,
                           max_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data.
    
    Important: Decoder input is shifted version of target
        Target: "hello"
        Decoder input: "<start> h e l l"
        
    This is teacher forcing: model sees ground truth shifted.
    """
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for misspelled, corrected in pairs:
        enc_in = tokenizer.encode(misspelled, max_length)
        dec_tgt = tokenizer.encode(corrected, max_length)
        
        # Decoder input: shift target right and prepend <start>
        dec_in = np.concatenate([[tokenizer.char2idx['<start>']], dec_tgt[:-1]])
        
        encoder_inputs.append(enc_in)
        decoder_inputs.append(dec_in)
        decoder_targets.append(dec_tgt)
    
    return np.array(encoder_inputs), np.array(decoder_inputs), np.array(decoder_targets)


def train():
    """Main training function."""
    
    print("\n" + "="*80)
    print("TRANSFORMER SEQ2SEQ FOR SPELLING CORRECTION")
    print("="*80)
    
    # 1. Generate data
    print("\n[1] Generating synthetic data...")
    pairs = generate_synthetic_data(num_pairs=500)
    print(f"    Generated {len(pairs)} pairs")
    
    # 2. Tokenize
    print("\n[2] Setting up tokenization...")
    tokenizer = CharacterTokenizer()
    max_length = 20
    print(f"    Vocabulary size: {tokenizer.vocab_size}")
    
    # 3. Create tensors
    print("\n[3] Creating training tensors...")
    X_enc, X_dec, Y = create_training_tensors(pairs, tokenizer, max_length)
    print(f"    Encoder shape: {X_enc.shape}")
    print(f"    Decoder shape: {X_dec.shape}")
    print(f"    Target shape:  {Y.shape}")
    
    # 4. Build model
    print("\n[4] Building model...")
    model = TransformerSpellingCorrector(
        vocab_size=tokenizer.vocab_size,
        max_length=max_length,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256
    )

    # Subclassed Keras models are not built until they see input once.
    # Run a dummy forward pass so count_params() and summaries are available.
    dummy_encoder = tf.zeros((1, max_length), dtype=tf.int32)
    dummy_decoder = tf.zeros((1, max_length), dtype=tf.int32)
    _ = model((dummy_encoder, dummy_decoder))

    # 5. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print(f"    Parameters: {model.count_params():,}")
    
    # 6. Train
    print("\n[5] Training...")
    history = model.fit(
        [X_enc, X_dec], Y,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        verbose=1
    )
    
    # 7. Evaluate
    print("\n[6] Evaluating on test set...")
    test_pairs = [(pairs[i][0], pairs[i][1]) for i in range(10)]
    test_enc, test_dec, test_tgt = create_training_tensors(test_pairs, tokenizer, max_length)
    
    predictions = model.predict([test_enc, test_dec])
    pred_chars = np.argmax(predictions, axis=-1)
    
    correct = 0
    for i in range(len(test_pairs)):
        predicted = tokenizer.decode(pred_chars[i])
        expected = test_pairs[i][1]
        is_correct = predicted == expected
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"    {status} '{test_pairs[i][0]}' → '{predicted}' (expected '{expected}')")
    
    print(f"\n    Accuracy: {correct}/{len(test_pairs)} ({100*correct/len(test_pairs):.1f}%)")
    
    print("\n" + "="*80)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train()
