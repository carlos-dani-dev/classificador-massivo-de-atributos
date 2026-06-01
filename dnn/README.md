## dnn
<p>
Nesta pasta estruturamos a rede neural descrita no paper.<br>
A rede neural densa do classificador massivo deve ser definida separadamente para cada base de imagens.
</p>

### O processo de treinamento
<p>
A rede neural multi-tarefa definida para cada base de imagens pode variar em relação à função de perda e formato de saída a depender da quantidade de atributos<br>
com que cada imagem da base de treinamento é anotada e também da quantidade de classes com as quais cada atributo é categorizado.
</p>

### Outputs imediatos do treinamento
<p>
Para cada treinamento, precisamos definir hiperparâmetros como <b><i>initial learning rate</i></b>, <b><i>epochs</i></b> e <b><i>batch size</i></b>.<br>
Uma vez que o script python <b><i>mac_{dataset_name}.py</i></b> termina sua execução, são gerados automaticamente:
<ul>
	<li>- history de treinamento em formato csv</li>
	<li>- os rótulos codificados da saída do modelo (label_encoders.pkl)</li>
	<li>- o modelo .keras, pronto para carregamento</li>
</ul>
</p>

### Fluxo visual de execução
