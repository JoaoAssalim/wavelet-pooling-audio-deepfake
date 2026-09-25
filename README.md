# Custom CNN Audio DeepFake Detection — v2.5

Versão oficial e final do projeto de mestrado para detecção de áudio deepfake com CNNs em PyTorch.

Os áudios de 16 kHz do **ASVspoof 2019 Logical Access (LA)** são ajustados para 4 segundos e convertidos em representações cepstrais **MFCC** e **LFCC**. O treino usa os splits train/dev do ASVspoof 2019. A avaliação é feita no eval do **ASVspoof 2019 LA** e no eval do **ASVspoof 2021 LA**.

Três arquiteturas classificam cada amostra como **bonafide** ou **spoof**:

- CNN com `MaxPool2d`
- CNN com `AvgPool2d`
- CNN com pooling wavelet na banda de aproximação (baixa frequência), com SpecAugment no treino

Cada arquitetura é treinada em **5 seeds**: `42`, `49`, `51`, `69` e `73`.

## Estrutura

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── normalize_data.ipynb
│   ├── analyze_results.ipynb
│   ├── max_pooling_seeds/
│   ├── average_pooling_seeds/
│   ├── wavelet_pooling_seeds/
│   ├── v1/                            # Experimentos anteriores
│   ├── v2/                            # Experimentos anteriores
│   └── v2.1/
├── results/
│   └── v2.5/                          # Resultados oficiais
└── (pastas locais, não versionadas)
    └── Data/
        ├── *_features.npy
        ├── *_meta.csv
        ├── DS_10283_3336/             # ASVspoof 2019 LA
        └── ASVspoof2021_LA_eval/      # ASVspoof 2021 LA eval
```

Os notebooks oficiais da v2.5 são:

| Pasta | Notebooks | Modelo |
| --- | --- | --- |
| `notebooks/max_pooling_seeds/` | `cnn_max_pooling_{42,49,51,69,73}.ipynb` | CNN com `MaxPool2d` |
| `notebooks/average_pooling_seeds/` | `cnn_average_pooling_{42,49,51,69,73}.ipynb` | CNN com `AvgPool2d` |
| `notebooks/wavelet_pooling_seeds/` | `custom_wavelet_cnn_model_low_freq_mask_seed_{42,49,51,69,73}.ipynb` | CNN com pooling wavelet (cA) e SpecAugment |

As pastas `notebooks/v1/`, `notebooks/v2/` e `notebooks/v2.1/` guardam apenas experimentos anteriores.

## Requisitos

- Python 3.12 ou 3.13
- pip
- Espaço em disco para o ASVspoof 2019 LA, o eval do ASVspoof 2021 LA e a pasta `Data/` (vários GB)
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

## Dataset

O pré-processamento espera o ASVspoof 2019 LA em:

```text
Data/DS_10283_3336/LA/
```

Baixe o conjunto **ASVspoof 2019** no [Edinburgh DataShare (DS_10283_3336)](https://datashare.ed.ac.uk/handle/10283/3336) e extraia o conteúdo em `Data/`, de forma que a árvore fique assim:

```text
Data/DS_10283_3336/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
└── ASVspoof2019_LA_eval/flac/
```

A avaliação cruzada no ASVspoof 2021 espera:

```text
Data/ASVspoof2021_LA_eval/
├── flac/
└── trial_metadata.txt
```

Se o caminho do ASVspoof 2019 for diferente, altere `base_dataset_path` em `notebooks/normalize_data.ipynb`.

## Data

A pasta `Data/` **não entra no repositório** (está no `.gitignore`). Os arquivos de features podem ocupar vários GB e passam do limite de push do GitHub.

Ela é criada ao rodar `notebooks/normalize_data.ipynb`:

```text
Data/
├── train_features.npy
├── train_meta.csv
├── dev_features.npy
├── dev_meta.csv
├── eval_features.npy
├── eval_meta.csv
├── eval_2021_features.npy
└── eval_2021_meta.csv
```

Cada `*_features.npy` armazena MFCC e LFCC em `float16`, com shape `(amostras, 2, 20, 401)`. Os arquivos `*_meta.csv` armazenam filename, speaker, tipo de ataque e a classe `is_bonafide`.

Os notebooks de treino da v2.5 carregam as features com memory mapping a partir de `../../Data/`, convertem os dois canais cepstrais para uma entrada de 40 × 401 e normalizam train, dev, eval 2019 e eval 2021 com as estatísticas do conjunto de treino.

## Rodar os notebooks

Com o ambiente virtual ativo:

```bash
jupyter notebook
```

Ou, no VS Code / Cursor, abra a pasta `notebooks/` e selecione o kernel `venv`.

Os caminhos relativos dependem da pasta do notebook:

- `notebooks/normalize_data.ipynb` e `notebooks/analyze_results.ipynb` usam `../Data/` e `../results/v2.5/`
- os notebooks em `*_pooling_seeds/` usam `../../Data/` e `../../results/v2.5/`

Se o working directory for a raiz do projeto, os acessos aos arquivos vão falhar.

### Ordem de execução

1. **Pré-processamento (obrigatório se `Data/` estiver vazio)**
   - `notebooks/normalize_data.ipynb`
   - Lê os FLAC e os protocols do ASVspoof 2019 e do eval 2021.
   - Ajusta cada áudio de 16 kHz para exatamente 4 s, repetindo sinais curtos e truncando sinais longos.
   - Extrai 20 coeficientes MFCC e 20 coeficientes LFCC em 401 frames.
   - Grava `Data/{train,dev,eval,eval_2021}_features.npy` e os `*_meta.csv` correspondentes.

2. **Treino e avaliação (um notebook por seed)**  
   Depois dos arquivos em `Data/` existirem, rode os notebooks das pastas oficiais. Todos usam até 100 épocas, `AdamW`, redução da learning rate e early stopping baseado no EER de desenvolvimento.

   | Pasta | Batch size | Seeds |
   | --- | --- | --- |
   | `max_pooling_seeds/` | 128 | 42, 49, 51, 69, 73 |
   | `average_pooling_seeds/` | 128 | 42, 49, 51, 69, 73 |
   | `wavelet_pooling_seeds/` | 32 | 42, 49, 51, 69, 73 |

   O notebook wavelet executa as famílias `db4`, `db6`, `db8`, `db14`, `db16`, `db18`, `db24`, `db26`, `db28`, `db34`, `db36` e `db38`.

   Cada seed grava um conjunto `v1.x` em `results/v2.5/`:

   | Seed | Saída |
   | --- | --- |
   | 42 | `v1.1` |
   | 51 | `v1.2` |
   | 73 | `v1.3` |
   | 69 | `v1.4` |
   | 49 | `v1.5` |

3. **Análise dos resultados**
   - `notebooks/analyze_results.ipynb`
   - Agrega as 5 seeds de Max, Average e das famílias wavelet.
   - Calcula média e desvio-padrão do EER no ASVspoof 2019 e no ASVspoof 2021.
   - Grava os resumos em `results/v2.5/final_results_five_seeds_2019.csv` e `results/v2.5/final_results_five_seeds_2021.csv`.

Execute as células em ordem, de cima para baixo.

## Resultados gerados

```text
results/v2.5/
├── max/
│   ├── training_v1.x.csv
│   ├── 2019_max_results_v1.x.csv
│   └── 2021_max_results_v1.x.csv
├── average/
│   ├── training_v1.x.csv
│   ├── 2019_average_results_v1.x.csv
│   └── 2021_average_results_v1.x.csv
├── wavelets/
│   └── wavelet_results_v1.x/
│       ├── 2019_wavelet_results.csv
│       ├── 2021_wavelet_results.csv
│       └── training_<wavelet>.csv
├── final_results_five_seeds_2019.csv
└── final_results_five_seeds_2021.csv
```

- Os CSVs `2019_*` e `2021_*` guardam as métricas de avaliação de cada seed.
- Os arquivos `training_*.csv` armazenam loss, EER e learning rate por época.
- Os arquivos `final_results_five_seeds_*.csv` consolidam média e desvio-padrão do EER nas 5 seeds.

## Dependências principais

| Pacote | Uso |
| --- | --- |
| `torch` | Modelos, treino e DataLoader |
| `librosa` | Leitura de áudio, STFT e extração de MFCC/LFCC |
| `spafe` | Construção do banco de filtros lineares usado no LFCC |
| `ptwt` | Transformada wavelet 1D no pooling customizado |
| `scikit-learn` | Métricas de classificação |
| `pandas` | Metadados, históricos e resultados em CSV |
| `matplotlib` | Gráficos de loss, EER e ROC |
