# tests/test_core_components.py
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mi_experiments.core.cache import BatchedKVCache
from mi_experiments.utils.control import ControlVector
from mi_experiments.core.model import ModelArgs, Model
from mi_experiments.core.attention import Attention
from mi_experiments.core.rope import RoPE
from mi_experiments.inference.pipeline import GenerationPipeline
from mi_experiments.utils.loading import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mi_experiments.core.quantization import QuantizedLinear


def test_model_args_initialization():
    """Test ModelArgs initialization and post-init behavior"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000
    )
    assert args.num_key_value_heads == args.num_attention_heads
    assert args.head_dim is None
    assert args.tie_word_embeddings is True

def test_kv_cache():
    """Test KV Cache functionality"""
    head_dim = 64
    n_kv_heads = 8
    batch_size = 2
    seq_len = 10
    
    cache = BatchedKVCache(head_dim, n_kv_heads, batch_size)
    
    # Create sample keys and values
    keys = mx.random.normal((batch_size, n_kv_heads, seq_len, head_dim))
    values = mx.random.normal((batch_size, n_kv_heads, seq_len, head_dim))
    
    # Test update and fetch
    cached_keys, cached_values = cache.update_and_fetch(keys, values)
    assert cached_keys.shape == keys.shape
    assert cached_values.shape == values.shape
    
    # Test incremental updates
    new_keys = mx.random.normal((batch_size, n_kv_heads, 5, head_dim))
    new_values = mx.random.normal((batch_size, n_kv_heads, 5, head_dim))
    
    updated_keys, updated_values = cache.update_and_fetch(new_keys, new_values)
    assert updated_keys.shape == (batch_size, n_kv_heads, seq_len + 5, head_dim)

def test_control_vector():
    """Test ControlVector functionality"""
    directions = {
        0: mx.array([1.0, 2.0, 3.0]),
        1: mx.array([4.0, 5.0, 6.0])
    }
    
    cv = ControlVector(
        model_type="llama",
        directions=directions
    )
    
    assert cv.model_type == "llama"
    assert len(cv.directions) == 2
    assert mx.array_equal(cv.directions[0], mx.array([1.0, 2.0, 3.0]))

@pytest.fixture(scope="session")
def pipeline():
    """Fixture for generation pipeline"""
    model_path = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    model, tokenizer = load(model_path)
    return GenerationPipeline(model, tokenizer)

def test_basic_inference(pipeline):
    """Test basic model inference"""
    prompts = ["Hello, how are you?"]
    
    # Collect all responses from generator
    responses = list(pipeline.generate(
        prompts, 
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9
    ))
    
    # Get final response from last generation step
    final_responses = responses[-1] if responses else []
    assert len(final_responses) == len(prompts)
    assert isinstance(final_responses[0], str)

def test_batched_inference(pipeline):
    """Test batched inference"""
    prompts = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?"
    ]
    
    responses = pipeline.batch_generate(
        prompts,
        batch_size=2,
        max_new_tokens=10,
        temperature=0.0  # Deterministic for testing
    )
    
    assert len(responses) == len(prompts)
    for response in responses:
        assert isinstance(response, str)

def test_streamed_inference(pipeline):
    """Test streaming generation"""
    prompt = "Tell me a story"
    pipeline.stream = True
    
    response_generator = pipeline.generate(prompt, max_new_tokens=10)
    assert hasattr(response_generator, '__iter__')
    
    responses = list(response_generator)
    assert len(responses) > 0
    assert all(isinstance(r, list) for r in responses)
    assert all(isinstance(r[0], str) for r in responses)

def test_control_vector_inference(pipeline):
    """Test inference with control vector"""
    control_vector = ControlVector.import_gguf("test_control_vector.gguf")
    prompts = ["Tell me a happy story"]
    
    # Generate with control vector
    responses_with_control = list(pipeline.generate(
        prompts,
        max_new_tokens=10,
        emphasis_config={"control_vector": control_vector, "control_strength": 1.0}
    ))
    
    # Generate without control vector 
    responses_without_control = list(pipeline.generate(
        prompts,
        max_new_tokens=10
    ))
    
    # Compare final responses
    final_with_control = responses_with_control[-1] if responses_with_control else []
    final_without_control = responses_without_control[-1] if responses_without_control else []
    
    assert len(final_with_control) == len(final_without_control)
    assert final_with_control != final_without_control

def test_attention_mechanism():
    """Test attention mechanism"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000,
        head_dim=64
    )
    
    attention = Attention(args)
    
    # Test input
    batch_size = 2
    seq_len = 10
    hidden_size = args.hidden_size
    
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Test without cache
    output = attention(x)
    assert output.shape == x.shape
    
    # Test with cache
    cache = BatchedKVCache(args.head_dim, args.num_key_value_heads, batch_size)
    output_with_cache = attention(x, cache=cache)
    assert output_with_cache.shape == x.shape

def test_end_to_end_inference(pipeline):
    """Test complete inference pipeline"""
    prompt = "Explain how a car engine works"
    responses = list(pipeline.generate([prompt]))
    final_response = responses[-1][0][0] if responses else ""
    
    assert isinstance(final_response, str)
    assert len(final_response) > 0
    
    # Test with control vector
    cv = ControlVector.import_gguf("test_control_vector.gguf")
    
    controlled_responses = list(pipeline.generate(
        [prompt],
        emphasis_config={"control_vector": cv, "control_strength": 1.0}
    ))
    final_controlled = controlled_responses[-1][0][0] if controlled_responses else ""
    
    assert isinstance(final_controlled, str)
    assert len(final_controlled) > 0
    assert final_controlled != final_response

def test_rope_embeddings():
    """Test rotary positional embeddings"""
    dims = 64
    max_position_embeddings = 2048
    
    rope = RoPE(
        dims=dims,
        max_position_embeddings=max_position_embeddings,
        base=10000,
        scale=1.0
    )
    
    # Test input
    batch_size = 2
    seq_len = 10
    num_heads = 8
    x = mx.random.normal((batch_size, num_heads, seq_len, dims))
    
    # Test regular embedding
    output = rope(x)
    assert output.shape == x.shape
    
    # Test with offset
    output_with_offset = rope(x, offset=5)
    assert output_with_offset.shape == x.shape

def test_layer_scaling():
    """Test layer-level scaling"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000
    )
    
    model = Model(args)
    
    # Test layer scaling configuration
    emphasis_config = {
        "layers": {
            0: 1.5,  # Scale first layer by 1.5x
            2: 0.5   # Scale third layer by 0.5x
        }
    }
    
    model.set_emphasis_config(emphasis_config)
    
    # Verify layer scales
    assert mx.array_equal(model.layers[0].layer_scale, mx.array(1.5))
    assert mx.array_equal(model.layers[1].layer_scale, mx.array(1.0))  # Default
    assert mx.array_equal(model.layers[2].layer_scale, mx.array(0.5))
    assert mx.array_equal(model.layers[3].layer_scale, mx.array(1.0))  # Default

def test_head_scaling():
    """Test attention head scaling"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000
    )
    
    model = Model(args)
    
    # Test head scaling configuration
    emphasis_config = {
        "heads": {
            0: {  # First layer
                1: 2.0,  # Scale second head by 2x
                3: 0.0   # Ablate fourth head
            },
            2: {  # Third layer
                0: 1.5,  # Scale first head by 1.5x
                7: 0.5   # Scale last head by 0.5x
            }
        }
    }
    
    model.set_emphasis_config(emphasis_config)
    
    # Verify head scales
    assert mx.array_equal(model.layers[0].self_attn.head_scale[1], mx.array(2.0))
    assert mx.array_equal(model.layers[0].self_attn.head_scale[3], mx.array(0.0))
    assert mx.array_equal(model.layers[2].self_attn.head_scale[0], mx.array(1.5))
    assert mx.array_equal(model.layers[2].self_attn.head_scale[7], mx.array(0.5))

def test_neuron_scaling():
    """Test MLP neuron scaling"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000
    )
    
    model = Model(args)
    
    # Test neuron scaling configuration
    emphasis_config = {
        "neurons": {
            1: {  # Second layer
                10: 1.5,   # Scale neuron 10 by 1.5x
                20: 0.0,   # Ablate neuron 20
                30: 0.5    # Scale neuron 30 by 0.5x
            }
        }
    }
    
    model.set_emphasis_config(emphasis_config)
    
    # Verify neuron scales
    assert mx.array_equal(model.layers[1].mlp.neuron_scale[10], mx.array(1.5))
    assert mx.array_equal(model.layers[1].mlp.neuron_scale[20], mx.array(0.0))
    assert mx.array_equal(model.layers[1].mlp.neuron_scale[30], mx.array(0.5))

def test_emphasis_config_validation():
    """Test emphasis configuration validation"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000
    )
    
    model = Model(args)
    
    # Test invalid layer index
    with pytest.raises(ValueError, match="Layer index .* exceeds model layers"):
        model.set_emphasis_config({
            "layers": {
                5: 1.5  # Invalid layer index (only 4 layers)
            }
        })
    
    # Test invalid head index
    with pytest.raises(ValueError, match="Head index .* exceeds model heads"):
        model.set_emphasis_config({
            "heads": {
                0: {
                    10: 1.5  # Invalid head index (only 8 heads)
                }
            }
        })
    
    # Test invalid neuron index
    with pytest.raises(ValueError, match="Neuron index .* exceeds hidden dimension"):
        model.set_emphasis_config({
            "neurons": {
                0: {
                    2000: 1.5  # Invalid neuron index (only 1024 neurons)
                }
            }
        })
    
    # Test invalid scaling values
    with pytest.raises(ValueError, match="Invalid scaling factor"):
        model.set_emphasis_config({
            "layers": {
                0: "invalid"  # Invalid scaling value type
            }
        })

def test_combined_emphasis():
    """Test combined layer, head, and neuron scaling"""
    args = ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=1024,
        num_attention_heads=8,
        rms_norm_eps=1e-6,
        vocab_size=32000
    )
    
    model = Model(args)
    
    # Test combined scaling configuration
    emphasis_config = {
        "layers": {
            0: 1.5
        },
        "heads": {
            0: {
                1: 2.0
            }
        },
        "neurons": {
            0: {
                10: 1.5
            }
        }
    }
    
    model.set_emphasis_config(emphasis_config)
    
    # Generate sample input
    batch_size = 2
    seq_len = 10
    input_ids = mx.random.randint(0, args.vocab_size, (batch_size, seq_len))
    
    # Verify forward pass works with scaling
    output = model(input_ids)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert output[0].shape == (batch_size, args.vocab_size)  # logits
    assert isinstance(output[1], list)  # hidden states

def test_quantization_dequantization():
    """Test quantization and dequantization of weights"""
    # Create sample weight matrix
    weight = mx.random.normal((1024, 4096))
    
    # Quantize weights
    group_size = 64
    bits = 4
    scales, zeros, qweight = quantize(weight, group_size, bits)
    
    # Create quantized linear layer
    qlayer = QuantizedLinear(qweight, None, group_size, bits)
    qlayer.scales = scales
    qlayer.zeros = zeros
    
    # Test forward pass
    x = mx.random.normal((1, 1024))
    out_quantized = qlayer(x)
    
    # Compare with non-quantized result
    out_full = mx.linear(x, weight)
    
    # Check that results are close (allowing for quantization error)
    assert mx.allclose(out_quantized, out_full, rtol=0.1, atol=0.1)

def test_memory_usage():
    """Test memory usage during dequantization"""
    import psutil
    process = psutil.Process()
    
    # Get initial memory
    mem_start = process.memory_info().rss
    
    # Create large weight matrix
    weight = mx.random.normal((4096, 4096))
    
    # Quantize
    scales, zeros, qweight = quantize(weight, 64, 4)
    qlayer = QuantizedLinear(qweight, None, 64, 4)
    qlayer.scales = scales
    qlayer.zeros = zeros
    
    # Run forward pass
    x = mx.random.normal((1, 4096))
    _ = qlayer(x)
    
    # Check peak memory
    mem_peak = process.memory_info().rss
    
    # Memory increase should be less than full precision
    assert (mem_peak - mem_start) < (4096 * 4096 * 4)  # 4-bit vs 32-bit