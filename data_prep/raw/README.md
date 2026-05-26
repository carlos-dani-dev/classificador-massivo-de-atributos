# Processamento de imagens 'cruas'

<p>Os 2 datasets escolhidos possuem estruturas de diretório próprias, veja:</p>

```text
raw/
├── raw_celeba/
│   ├── pre_insight_celeba.ipynb
│   ├── loader_celeba.py
│   ├── list_attr_celeba.csv/
│   ├── img_align_celeba/
│   │   ├── img_align_celeba/
│   │       ├── 000001.jpg
│   │       ├── 000002.jpg
│   │       ├── ...
|   |
|   |
├── raw_casual/
│   ├── pre_insight_casual.ipynb
│   ├── loader_casual.py
│   ├── video_att_mapping_cc_dataset.csv/
│   ├── dataset/
│   │   ├── casual_pt1/
│   │       ├── 0000
|   |           ├── 0000_portuguese_nonscripted__1_raw_frame00000381.jpg
|   |           ├── ...
│   │       ├── 0001
|   |           ├── 0001_portuguese_nonscripted__1_raw_frame00000980.jpg
|   |           ├── ...
│   │       ├── ...
│   │   ├── casual_pt2/
│   │       ├── 1114
│   │       ├── 1115
│   │       ├── ...
│   │   ├── casual_pt3/
│   │       ├── 2228
│   │       ├── 2229
│   │       ├── ...
│   │   ├── casual_pt4/
│   │       ├── 3341
│   │       ├── 3342
│   │       ├── ...
│   │   ├── casual_pt5/
│   │       ├── 4454
│   │       ├── 4455
│   │       ├── ...

```

### Fluxo de execução

<p>Sobre cada dataset, o arquivo loader_{dataset_name}.py realiza, em ordem, as seguintes operações:<br>
    <ul>
        <li>1. redimensionamento da imagem.</li>
        <li>2. extração e recorte da face, via MTCNN</li>
        <li>3. vetorização da face, via FaceNet</li>
    </ul>
</p>
<p>O resultado é um dataset que carrega as anotações de cada imagem e seu respectivo vetor de embedding.<br>
O arquivo é salvo em formato pickle (.pkl), dentro de /proc/proc_{dataset_name}/
</p>