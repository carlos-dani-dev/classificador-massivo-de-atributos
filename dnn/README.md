# Treinamento do classificador
<p>
Nesta pasta estruturamos a rede neural descrita no paper.<br>
A rede neural densa do classificador massivo deve ser definida separadamente para cada base de imagens.
</p>

---

## O processo de treinamento
<p>
A rede neural multi-tarefa definida para cada base de imagens pode variar em relação à função de perda e formato de saída a depender da quantidade de atributos
com que cada imagem da base de treinamento é anotada e também da quantidade de classes com as quais cada atributo é categorizado.<br>
Por exemplo, se cada imagem da base de imagens A for anotada com 10 atributos binários, o classificador precisa usar funções de perda compatíveis, além de uma função de ativação sigmoid na ponta de cada sub-rede.
Agora, se em uma base de imagens B, cada imagem for anotada com 3 atributos multi-classe, o classificador provavelmente precisará de funções de perda categóricas, além de funções de ativação softmax na ponta de cada sub-rede.
</p>

---

## Outputs imediatos do treinamento
<p>
Para cada treinamento, precisamos definir hiperparâmetros como <b><i>initial learning rate</i></b>, <b><i>epochs</i></b> e <b><i>batch size</i></b>.<br>
Uma vez que o script python <b><i>mac_{dataset_name}.py</i></b> termina sua execução, são gerados automaticamente:
<ul>
	<li>history de treinamento em formato csv</li>
	<li>os rótulos codificados da saída do modelo (label_encoders.pkl)</li>
	<li>o modelo .keras, pronto para carregamento</li>
</ul>
</p>

---

## As instâncias de classficiadores
<p>
São 3 as instâncias de classificadores definidas durante a pesquisa:
</p>
- <b>mac_celeba</b>: MAC treinado com o CelebA Dataset balanceado sobre o atributo <b>Male</b>
- <b>mac_casual</b>: MAC treinado com o Casual Conversations v2 balanceado sobre o atributo <b>fitz_type</b>
- <b>mac_hybrid</b>: MAC treinado com o superdataset, balanceado, resultante da anotação massiva do Casual Conversations v2 com o mac_celeba
<p><b>O mac_hybrid</b> é naturalmente a versão mais completa do MAC, uma vez que foi treinada sobre um dataset anotado com os 2 atributos: Male e fitz_type</p>
