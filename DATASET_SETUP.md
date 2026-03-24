# How to obtain the data

## InfiniteBench

Source Paper: [InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718).

*   **Download**: Get `longbook_choice_eng.jsonl` from the [HuggingFace Repository](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/main).
*   **Placement**: Save the file to:
    ```
    ./data/InfiniteBench/longbook_choice_eng.jsonl
    ```


## NovelQA

Source Paper: [NovelQA: Benchmarking Question Answering on Documents Exceeding 200K Tokens](https://arxiv.org/abs/2403.12766).

*   **Availability**: Partial dataset available openly. Contact the authors of the paper for full restricted data.
*   **Setup Instructions**:
    Run the following commands to clone the available data into the correct directory:
    ```bash
    pip install huggingface_hub
    huggingface-cli login
    cd data
    git clone https://huggingface.co/datasets/NovelQA/NovelQA
    ```
```

