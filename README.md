## classificador-massivo-de-atributos
<p>
Esse repositório implementa fielmente o classificador de atributos faciais massivo idealizado por Therhörst,<br>
em <a href="https://arxiv.org/abs/2012.01030">MAAD-Face: A Massive Annotated Attribute Dataset for face images</a>.
</p>

<p><br></p>

### Estrutura do código
<p>
O projeto é subdividido em 2 pastas principais, com funções diferentes:<br>
<b>data_prep</b>: para vetorização e balanceamento das bases de imagens selecionadas para treinamento do MAC.<br>
<b>dnn</b>: onde a rede neural densa, dnn, é definida. Nesta pasta, o MAC é treinado individualmente com cada
uma das bases de dados selecionadas.
</p>

