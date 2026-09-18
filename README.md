# Custom CNN Audio DeepFake Detection — v2

Projeto de mestrado para detecção de áudio deepfake com CNNs em PyTorch. Os áudios de 16 kHz do **ASVspoof 2019 Logical Access (LA)** são ajustados para 4 segundos e convertidos em representações cepstrais **MFCC** e **LFCC**. Em seguida, três arquiteturas de CNN classificam cada amostra como **bonafide** ou **spoof**: pooling máximo, pooling médio e pooling wavelet pela sub-banda de maior energia.

## Estrutura atual do repositório

```text
.
├── .gitignore                         # Ignora Data/ e DS_10283_3336/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── normalize_data.ipynb
│   ├── cnn_max_pooling.ipynb
│   ├── cnn_average_pooling.ipynb
│   ├── custom_wavelet_cnn_model_highest_energy.ipynb
│   ├── custom_wavelet_cnn_model_highest_energy_v2.ipynb
│   ├── analyze_results.ipynb
│   └── v1/                           # Experimentos anteriores e variações de versão
│       ├── normalize_data.ipynb
│       ├── simple_cnn_model_max_pooling.ipynb
│       ├── simple_cnn_model_average_pooling.ipynb
│       ├── custom_wavelet_cnn_model_highest_energy.ipynb
│       ├── custom_wavelet_cnn_model_highest_energy_v2.ipynb
│       ├── custom_wavelet_cnn_model_adaptative.ipynb
│       ├── different_wavelets/
│       └── different_seeds/
├── results/
│   └── v2/
│       ├── max_results.csv
│       ├── average_results.csv
│       ├── wavelet_results.csv
│       ├── training_max.csv
│       ├── training_average.csv
│       └── training_db*.csv
└── (pastas locais criadas em execução)
    ├── Data/                          # Features e metadados gerados localmente
    └── DS_10283_3336/                 # Dataset ASVspoof 2019 LA não versionado
```

As pastas `Data/` e `DS_10283_3336/` não ficam no Git e são criadas localmente quando você baixa o dataset e executa o pré-processamento.

## Requisitos

- Python 3.12
- pip
- Espaço em disco para o ASVspoof 2019 LA e para a pasta `Data/` (vários GB)
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

O `pandas` está incluído no `requirements.txt`. Para executar os notebooks fora do VS Code/Cursor, instale também o Jupyter:

```bash
pip install jupyter
```

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

## Data

A pasta `Data/` **não entra no repositório** (está no `.gitignore`). Os arquivos de features podem ocupar vários GB e passam do limite de push do GitHub.

Ela é criada ao rodar `notebooks/normalize_data.ipynb` e fica na raiz do projeto:

```text
Data/
├── train_features.npy
├── train_meta.csv
├── dev_features.npy
├── dev_meta.csv
├── eval_features.npy
└── eval_meta.csv
```

Cada `*_features.npy` armazena MFCC e LFCC em `float16`, com shape `(amostras, 2, 20, 401)`. Os arquivos `*_meta.csv` armazenam filename, speaker, tipo de ataque e a classe `is_bonafide`.

Os notebooks de treino carregam as features com memory mapping a partir de `../Data/`, convertem os dois canais cepstrais para uma entrada de 40 × 401 e normalizam train, dev e eval com as estatísticas do conjunto de treino. Sem esses arquivos, o pré-processamento precisa ser executado novamente.

## Rodar os notebooks

Com o ambiente virtual ativo:

```bash
jupyter notebook
```

Ou, no VS Code / Cursor, abra a pasta `notebooks/` e selecione o kernel `venv`.

Os caminhos relativos (`../Data/`, `../DS_10283_3336/` e `../results/v2/`) assumem que o notebook está sendo executado a partir de `notebooks/`. No Jupyter, abra os arquivos dessa pasta. Se o working directory for a raiz do projeto, os acessos aos arquivos vão falhar.

### Ordem de execução

1. **Pré-processamento (obrigatório se `Data/` estiver vazio)**
   - `notebooks/normalize_data.ipynb`
   - Lê os FLAC e os protocols.
   - Ajusta cada áudio de 16 kHz para exatamente 4 s, repetindo sinais curtos e truncando sinais longos.
   - Extrai 20 coeficientes MFCC e 20 coeficientes LFCC em 401 frames.
   - Grava `Data/{train,dev,eval}_features.npy` e `Data/{train,dev,eval}_meta.csv`.
   - Esta etapa é a mais lenta (percorre train, dev e eval).

2. **Treino e avaliação (independentes entre si)**  
   Depois dos arquivos em `Data/` existirem, rode qualquer um destes notebooks. Todos usam seed 42, batch size 32, até 100 épocas, `AdamW`, redução da learning rate e early stopping baseado no EER de desenvolvimento.

| Notebook | Modelo |
| --- | --- |
| `cnn_max_pooling.ipynb` | CNN de referência com `MaxPool2d` |
| `cnn_average_pooling.ipynb` | CNN de referência com `AvgPool2d` |
| `custom_wavelet_cnn_model_highest_energy_v2.ipynb` | CNN com pooling wavelet pela sub-banda de maior energia |

O notebook wavelet executa as famílias `db4`, `db6`, `db8`, `db14`, `db16`, `db18`, `db24`, `db26`, `db28`, `db34`, `db36` e `db38`.

3. **Análise dos resultados**
   - `notebooks/analyze_results.ipynb`
   - Carrega os CSVs em `results/v2/`.
   - Compara EER, ROC AUC e demais métricas entre Max, Average e as famílias wavelet.
   - Visualiza também os históricos de loss e EER dos treinamentos.

Execute as células em ordem, de cima para baixo.

## Resultados gerados

Os notebooks de treino gravam os resultados da versão atual em:

```text
results/v2/
├── max_results.csv
├── average_results.csv
├── wavelet_results.csv
├── training_max.csv
├── training_average.csv
└── training_<wavelet>.csv
```

- `max_results.csv`, `average_results.csv` e `wavelet_results.csv` contêm loss de avaliação, EER, threshold do EER, accuracy, precision, recall, F1, matriz de confusão e ROC AUC.
- Os arquivos `training_*.csv` armazenam loss, EER e learning rate por época.

## Dependências principais

| Pacote | Uso |
| --- | --- |
| `torch` | Modelos, treino e DataLoader |
| `librosa` | Leitura de áudio, STFT e extração de MFCC/LFCC |
| `spafe` | Construção do banco de filtros lineares usado no LFCC |
| `ptwt` | Transformada wavelet 2D nos poolings customizados |
| `scikit-learn` | Métricas de classificação |
| `pandas` | Metadados, históricos e resultados em CSV |
| `matplotlib` | Gráficos de loss, EER e ROC |
