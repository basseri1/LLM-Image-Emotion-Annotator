# LLM Image Emotion Annotator (Arabic)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8--3.12-blue)
![Last Commit](https://img.shields.io/github/last-commit/basseri1/LLM-Image-Emotion-Annotator)
![Emoji](https://img.shields.io/badge/😃-Emotions-orange)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

Automate the annotation of emotions in images using state-of-the-art language models (OpenAI GPT-4o and Google Gemini 1.5 Pro) with zero-shot, few-shot, and chain-of-thought prompting. The tool outputs emotion labels in Arabic, based on Plutchik's wheel of emotions.

---

## Table of Contents

- [Emotion Labels](#emotion-labels)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output-results-csv)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Emotion Labels

| English        | Arabic    |
|---------------|-----------|
| Joy           | سعادة     |
| Trust         | ثقة       |
| Fear          | خوف       |
| Surprise      | مفاجأة    |
| Sadness       | حزن       |
| Disgust       | قرف       |
| Anger         | غضب       |
| Anticipation  | ترقب      |
| Neutral       | محايد     |

---

## Project Structure

```
images/                  # Input images to annotate
few_shot_examples/       # Three required few-shot example images (see below)
results/                 # Output CSV files
main.py                  # Main script
requirements.txt         # Python dependencies
utils/
  image_utils.py         # Image loading/encoding
  prompt_utils.py        # Prompt construction
  model_utils.py         # Model API interaction
.env                     # API keys and environment variables
```

---

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your_openai_api_key
     GOOGLE_API_KEY=your_google_api_key
     ```

3. **Add images:**
   - Place your images in the `images/` folder. Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP, ICO, HEIC (for HEIC, install optional dependencies).
   - For few-shot prompting, add three images to `few_shot_examples/`:
     - `sadness.*`   (for حزن)
     - `surprise.*`  (for مفاجأة)
     - `disgust.*`   (for قرف)
   - The extension does not matter (e.g., `sadness.png`, `sadness.jpg`, etc.).

---

## Usage

Run the main script:
```bash
python main.py
```

You will be prompted to select a temperature value for the models (see [Configuration](#configuration)).

---

## Configuration

**Temperature Setting:**  
- Controls the determinism/creativity of model responses.
- Lower values (0.0-0.3): More consistent, accurate labeling.
- Higher values (0.4-0.7): May help if models are too cautious/refusing.
- Very high values (0.8-1.0): Not recommended for classification.

---

## Output: Results CSV

- Results are saved in the `results/` folder with a timestamped filename (e.g., `results_20240610_153045.csv`).
- **CSV columns:**
  - `image_name`
  - `gpt4o_zero_shot`
  - `gpt4o_few_shot`
  - `gpt4o_cot`
  - `gpt4o_cot_reasoning`
  - `gemini_zero_shot`
  - `gemini_few_shot`
  - `gemini_cot`
  - `gemini_cot_reasoning`
  - `gpt4o_zero_shot_request`
  - `gpt4o_few_shot_request`
  - `gpt4o_cot_request`
  - `gemini_zero_shot_request`
  - `gemini_few_shot_request`
  - `gemini_cot_request`
  - `timestamp`

---

## Troubleshooting

- **Missing API Key:** Ensure your `.env` file is in the project root and contains valid keys.
- **Few-shot Example Error:** Make sure all three required few-shot images are present and named correctly.
- **Image Format Error:** For HEIC support, install `pillow-heif` and `pyheif`.
- **API Access:** Ensure your API keys have access to vision endpoints.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

MIT License

---

## Acknowledgments

- [OpenAI](https://openai.com/)
- [Google Gemini](https://ai.google.com/gemini/)
- [Plutchik's Wheel of Emotions](https://en.wikipedia.org/wiki/Robert_Plutchik)

