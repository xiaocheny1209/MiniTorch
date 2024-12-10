This project implemented the core components of Pytorch, including:

- Mathematical and computational foundations, including derivatives and backpropagation.
- Tensor operations, broadcasting, and gradients.
- Implementing parallelism and optimizing CUDA operations.
- Building image recognition systems, including LeNet for digit recognition and 1D convolutional models for NLP sentiment classification.
- Docs: https://minitorch.github.io/

Install the necessary dependencies:
```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
python -m pip install -Ue .
```

Test the codes
```bash
pytest