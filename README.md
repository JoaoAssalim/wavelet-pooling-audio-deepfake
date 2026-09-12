# Custom CNN Audio DeepFake Detection

Projeto de mestrado para detecção de áudio deepfake com CNNs em PyTorch. Os áudios do **ASVspoof 2019 Logical Access (LA)** são normalizados e convertidos em features (MFCC, spectral centroid, spectral rolloff, chroma STFT, ZCR e RMS). Em seguida, diferentes arquiteturas classificam cada amostra como **bonafide** ou **spoof**.

## Estrutura

```text
.
├── Data/                  # CSVs gerados localmente (não versionado; ver /Data)
├── DS_10283_3336/         # Dataset ASVspoof 2019 LA (não versionado)
├── notebooks/
│   ├── normalize_data.ipynb
│   ├── simple_cnn_model_max_pooling.ipynb
│   ├── simple_cnn_model_average_pooling.ipynb
│   ├── custom_wavelet_cnn_model_highest_energy.ipynb
│   ├── custom_wavelet_cnn_model_highest_energy_v2.ipynb
│   └── custom_wavelet_cnn_model_adaptative.ipynb
└── requirements.txt
```

## Requisitos

- Python 3.12
- pip
- Espaço em disco para o ASVspoof 2019 LA e para a pasta `/Data` (vários GB)
- GPU opcional: CUDA ou Apple Silicon (MPS). Sem GPU, o treino usa CPU.

## Instalação

Na raiz do repositório:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`pandas` e `jupyter` são usados pelos notebooks e não estão no `requirements.txt`.

## Dataset

Os notebooks esperam o ASVspoof 2019 LA em:

```text
DS_10283_3336/LA/
```

Baixe o conjunto **ASVspoof 2019** no [Edinburgh DataShare (DS_10283_3336)](https://datashare.ed.ac.uk/handle/10283/3336) e extraia o conteúdo na raiz do projeto, de forma que a árvore fique assim:

```text
DS_10283_3336/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
└── ASVspoof2019_LA_eval/flac/
```

Se o caminho for diferente, altere `base_dataset_path` em `notebooks/normalize_data.ipynb`.

## /Data

A pasta `/Data` **não entra no repositório** (está no `.gitignore`). Os CSVs somam vários GB e passam do limite de push do GitHub.

Ela é criada ao rodar `notebooks/normalize_data.ipynb` e fica na raiz do projeto:

```text
Data/
├── train_data_features.csv
├── dev_data_features.csv
└── eval_data_features.csv
```

Os notebooks de treino leem esses arquivos em `../Data/`. Sem eles, o pré-processamento precisa ser executado de novo.

## Rodar os notebooks

Com o ambiente virtual ativo:

```bash
jupyter notebook
```

Ou, no VS Code / Cursor, abra a pasta `notebooks/` e selecione o kernel `venv`.

Os caminhos relativos (`../Data/`, `../DS_10283_3336/`) assumem que o notebook está sendo executado a partir de `notebooks/`. No Jupyter, abra os arquivos dessa pasta. Se o working directory for a raiz do projeto, os `read_csv` e `to_csv` vão falhar.

### Ordem de execução

1. **Pré-processamento (obrigatório se `Data/` estiver vazio)**
   - `notebooks/normalize_data.ipynb`
   - Lê os FLAC + protocols, normaliza cada áudio para 4 s a 16 kHz e grava:
     - `Data/train_data_features.csv`
     - `Data/dev_data_features.csv`
     - `Data/eval_data_features.csv`
   - Esta etapa é a mais lenta (percorre train, dev e eval).

2. **Treino e avaliação (independentes entre si)**  
   Depois dos CSVs existirem, rode qualquer um destes notebooks. Cada um carrega os mesmos arquivos em `Data/`, treina por até 100 épocas com early stopping e reporta accuracy, precision, recall, F1 e curva ROC no conjunto de avaliação.

| Notebook | Modelo |
| --- | --- |
| `simple_cnn_model_max_pooling.ipynb` | CNN com `MaxPool2d` |
| `simple_cnn_model_average_pooling.ipynb` | CNN com `AvgPool2d` |
| `custom_wavelet_cnn_model_highest_energy.ipynb` | CNN com pooling wavelet (banda de maior energia) |
| `custom_wavelet_cnn_model_highest_energy_v2.ipynb` | Variante do pooling por maior energia |
| `custom_wavelet_cnn_model_adaptative.ipynb` | CNN com pooling wavelet adaptativo (fusão das 4 bandas) |

Execute as células em ordem, de cima para baixo.

## Dependências principais

| Pacote | Uso |
| --- | --- |
| `torch` | Modelos, treino e DataLoader |
| `librosa` | Leitura de áudio e extração de features |
| `ptwt` | Transformada wavelet 2D nos poolings customizados |
| `scikit-learn` | Métricas de classificação |
| `matplotlib` | Gráficos de loss, accuracy e ROC |
