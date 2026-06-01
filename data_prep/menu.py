import pandas as pd

df_celeba=pd.read_pickle("./proc/proc_celeba/embeddings_celeba.pkl")
atributos_celeba = df_celeba.columns[2:]

df_casual=pd.read_pickle("./proc/proc_casual/embeddings_casual.pkl")
atributos_casual = df_casual.columns[2:]

menu = """
    Selecione uma das bases de imagens cadastradas
    
    a) Celeb A
    b) Casual Conversations v2

    s) Sair

    Base de imagens: """

menu_dataset = """
    Selecione o que fazer com a base de imagens

    c) Verificar anotações
    d) Verificar tamanho
    e) Verificar balanceamento das anotações

    f) Balancear base de imagens
    
    s) Trocar base de imagens
    Opção: """

menu_over_undersampling = """
    Qual princípio de balanceamento você deseja utilizar?

    1) Oversampling
    2) Undersampling

    s) Sair
    
    Opção: """

menu_metodo_oversampling = """
    Selecione o método de oversampling sobre vetores que você deseja utilizar:

    a) SMOTE

    Opção: """

menu_metodo_undersampling = """
    Selecione o método de undersampling sobre vetores que você deseja utilizar:

    a) Tomek Link

    Opção: """


def tam_base(base):
    if base == "Celeb A":
        return len(df_celeba)
    elif base == "Casual Conversations v2":
        return len(df_casual)


def anotacoes(base):
    if base == "Celeb A":
        return [c for c in atributos_celeba]
    elif base == "Casual Conversations v2":
        return [c for c in atributos_casual]


def balanceamento_atributos(base):
    if base == "Celeb A":
        menu_atributos = "Atributos anotados: "

        for i in range(len(atributos_celeba)):
            menu_atributos += f"\n  {i}) {atributos_celeba[i]} {df_celeba[atributos_celeba[i]].unique()}"
        menu_atributos += "\n\n Nº do atributo: "

        print("\n===================\n")
        atributo = atributos_celeba[int(input(menu_atributos))]

        print("\n===================\n")
        print(f"Distribuição de classes do atributo {atributo} [{base}]:\n")
        print(f"{df_celeba[atributo].value_counts()}")
        print("\n===================\n")
    
    elif base == "Casual Conversations v2":
        menu_atributos = "Atributos anotados: "

        for i in range(len(atributos_casual)):
            menu_atributos += f"\n  {i}) {atributos_casual[i]} {df_casual[atributos_casual[i]].unique()}"
        menu_atributos += "\n\n Nº do atributo: "

        print("\n===================\n")
        atributo = atributos_casual[int(input(menu_atributos))]

        print("\n===================\n")
        print(f"Distribuição de classes do atributo {atributo} [{base}]:\n")
        print(f"{df_casual[atributo].value_counts()}")
        print("\n===================\n")


def aplicar_metodo_over(base, metodo_over):
    if metodo_over == "a": print(f"Aplicando SMOTE em {base}")


def aplicar_metodo_under(base, metodo_under):
    if metodo_under == "a": print(f"Aplicando Tomek Link em {base}")


def balancear_base(base):
    menu_atributos = "\n    Selecione o atributo pivô do balanceamento: "

    for i in range(len(atributos_casual)):
        menu_atributos += f"\n      {i}) {atributos_casual[i]} {df_casual[atributos_casual[i]].unique()}"
    menu_atributos += "\n\n     Nº do atributo pivô: "

    print("\n===================")
    atributo = atributos_casual[int(input(menu_atributos))]

    print("\n===================")
    principio = input(menu_over_undersampling)
    if principio == "1":
        print("\n===================")
        metodo_over = input(menu_metodo_oversampling)
        aplicar_metodo_over(base, metodo_over)
    elif principio == "2":
        print("\n===================")
        metodo_under = input(menu_metodo_undersampling)
        aplicar_metodo_under(base, metodo_under)


def menu_base(base):
    while(True):
        opt_ = input(menu_dataset)
        if opt_ == 's':
            print("\n===================")
            break
        elif opt_ == 'c':
            lista_anotacoes = anotacoes(base)
            print("\n===================\n")
            print(f"c) A base de imagens {base} possui as anotacoes:\n\n{lista_anotacoes}.")
            print("\n===================\n")
        elif opt_ == 'd':
            total = tam_base(base)
            print("\n===================\n")
            print(f"d) A base de imagens {base} possui {total} faces.")
            print("\n===================\n")
        elif opt_ == 'e':
            balanceamento_atributos(base)
        elif opt_ == 'f':
            balancear_base(base)


if __name__ == "__main__":

    while(True):
        opt = input(menu)
        if opt == 's': break

        if opt=='a':
            print("\n===================")
            menu_base("Celeb A")
        elif opt=='b':
            print("\n===================")
            menu_base("Casual Conversations v2")
