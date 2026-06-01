# Classificador Massivo de Atributos

Este reposítório implementa o classificador massivo de atriutos proposto no paper <a href="https://github.com/pterhoer/maad-face">MAAD-Face: A Massively Annotated Attribute Dataset for Face Images</a>.<br>
O trabalho proposto foi desenvolvido como projeto de pesquisa de Iniciação Científica CNPQ.

## Fluxo de execução
<p>
  O classificador de atributos é resultado de um pipeline bem definido:<br>
1º) Vetorização e balanceamento das bases de imagens pré-anotadas que treinarão a rede neural<br>
2º) Cross-labelling das bases de imagens pré-anotadas<br>
3º) Treinamento da versão mais robusta da rede neural com uma base de imagens unificada mais completa
</p>
<p>Cada uma das etapas pode ser verificada em sua respectiva subpastas do projeto.</p>

## Arquivos
```text
mac/
├── cross_labelling/
│   ├── casual_annotated.pkl
│   ├── celeba_annotated.pkl
│   └── celeba_casual.py
├── data_prep/
│   ├── proc/
│   │   ├── proc_casual/
│   │   │   ├── balancing_casual.py
│   │   │   ├── embeddings_casual.pkl
│   │   │   └── fitz_type_balanced_embeddings_casual.pkl
│   │   └── proc_celeba/
│   │       ├── balancing_celeba.py
│   │       ├── embeddings_celeba.pkl
│   │       └── male_balanced_embeddings_celeba.pkl
│   ├── raw/
│   │   ├── raw_casual/
│   │   │   ├── dataset/
│   │   │   ├── loader_casual.py
│   │   │	└── video_att_mapping_cc_dataset.csv
│   │   └── raw_celeba/
│   │   	├── img_align_celeba/
│   │       ├── list_attr_celeba.csv
│   │       └── loader_celeba.py
│   └── README.md    
├── dnn/
│   ├── mac_casual/
│   │   ├── label_encoders.pkl
│   │   ├── mac_casual.py
│   │   ├── modelo_cc_200_4096_1e-05.keras
│   │   └── modelo_cc_200_4096.csv
│   ├── mac_celeba/
│   │   ├── mac_celeba.py
│   │   ├── modelo_celebA_200_1024.csv
│   │   └── modelo_celebA_200_1024.keras
│   ├── mac_hybrid/
│   │   ├── mac_hybrid.py
│   │   ├── modelo_superdataset_200_4096_1e-05.keras
│   │   ├── modelo_superdataset_200_4096.csv
│   │   └── superdataset_label_encoders.pkl
│   └── README.md
└── README.md
```
