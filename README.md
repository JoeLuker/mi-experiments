# MI Experiments

## Setup

```bash
pip install -e .
```

## Project Structure

- `core/`: Core model implementation
- `utils/`: Utility functions and helpers
- `analysis/`: Analysis tools and visualizations
- `ui/`: Web interface components
- `inference/`: Inference and generation code
- `tests/`: Unit tests
- `notebooks/`: Example notebooks
- `docs/`: Documentation

## Usage

See `notebooks/model_demo.ipynb` for usage examples.

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
