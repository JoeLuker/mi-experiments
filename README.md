# MI Experiments

A research project exploring emphasis and ablation techniques in transformer models using MLX. This project combines ideas from parallel inference optimization and representation engineering techniques.

## Features

- Transformer model implementation with MLX
- Configurable emphasis/ablation for:
  - Layers
  - Attention heads
  - Neurons
- Top-p sampling with temperature control
- Structured logging
- Parallel inference optimizations

## Installation

```bash
git clone https://github.com/yourusername/resume-experiments.git
cd resume-experiments
pip install -r requirements.txt
```

## Quick Start

```python
from src.inference.generator import self_sufficient_inference
from src.models import load_model

# Load model
model, tokenizer = load_model("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Configure emphasis
emphasis_config = {
    'layers': {'0': 1.5, '1': 0.0},
    'heads': {'3': {'1': 2.0, '2': 0.0}},
    'neurons': {'4': {'15': 1.5, '30': 0.0}}
}

# Generate text
response = self_sufficient_inference(
    model=model,
    tokenizer=tokenizer,
    prompts=["Who are you?"],
    emphasis_config=emphasis_config,
    max_tokens=100,
    temp=0.7,
    top_p=0.9
)
```

## Configuration

### Emphasis Configuration

The emphasis configuration allows you to modify the behavior of specific model components:

```json
{
    "layers": {
        "0": 1.5,  // Amplify first layer
        "1": 0.0   // Ablate second layer
    },
    "heads": {
        "3": {     // Configure heads in layer 3
            "1": 2.0,  // Double attention head 1
            "2": 0.0   // Ablate attention head 2
        }
    },
    "neurons": {
        "4": {     // Configure neurons in layer 4
            "15": 1.5, // Amplify neuron 15
            "30": 0.0  // Ablate neuron 30
        }
    }
}
```

## Acknowledgments

This project builds upon several excellent open source works:

- [MLX ParaLLM](https://github.com/willccbb/mlx_parallm) by willccbb - Fast parallel LLM inference techniques
- [MLX Examples](https://github.com/ml-explore/mlx-examples) by Apple - Core MLX implementation and examples
- [repeng](https://github.com/vgel/repeng) by Theia Vogel - Representation engineering implementation
- Original representation engineering research by [Andy Zou et al.](https://github.com/andyzoujm/representation-engineering)

## Citation

If you use this work in academic research, please cite the following:

```bibtex
@misc{zou2024representation,
  title={Representation Engineering: A Top-Down Approach to AI Alignment}, 
  author={Andy Zou and Long Phan and Sarah Chen and James Campbell and Alejandro Escontrela and Ivan Evtimov},
  year={2024},
  eprint={2310.01405},
  archivePrefix={arXiv}
}

@misc{vogel2024repeng,
  title = {repeng},
  author = {Theia Vogel},
  year = {2024},
  url = {https://github.com/vgel/repeng/}
}

@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```

## License

MIT License - see LICENSE file for details