# Importando Libs e Packages
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Antes de dar continuidade, será preciso conhecer os dados, pois dessa forma
# será possível identificar se há valores duplicados, vazios, etc..

# Conectando com os Dados
dataset = pd.read_csv("src/Kidney_data.csv")

# Visualizando as 5 primeiras linhas
print(dataset.head())

# Verificando as dimensões do dataset (shape)
print(dataset.shape)

# Verificando informações adicionais do dataset
print(dataset.info())

# Verficando se há valores Missing (valores ausentes)
print(dataset.isnull().sum())

# Verificando se há valroes duplicados
print(dataset.duplicated().sum())

# Estatistica descritiva das variaveis
print(dataset.describe())

# Teve ou não doença nos rins?
print(dataset.classification.value_counts())

# Grafico para visualização da classificação
sns.countplot(dataset['classification'])
plt.show()

# Tabela de frequência da variavel rbc (globulos vermelhos)
print(dataset.rbc.value_counts())

# Grafico para visualização da qtd de globulos vermelhos
sns.countplot(dataset['rbc'])
plt.show()

# Tabela frequencia da variavel ba (bacteria)
print(dataset.ba.value_counts())

# Grafico da frequencia de bacterias dos pacientes
sns.countplot(dataset['ba'])
plt.show()

# Tabela de frequência da coluna pc (células de pus)
print(dataset.pc.value_counts())

# Gráfico da frequencia da variavel pc
sns.countplot(dataset['pc'])
plt.show()

# Gráfico de frequencia das idades
print(dataset.age.value_counts())

# Gráfico da frequnecia das idade
sns.histplot(dataset['age'], bins=20, kde=True)
plt.show()

# Entrando na parte de processamento dos dados

# Tratar valores Missing
# Elimistar registros duplicados
# Transformação para deixar os valores na mesma escala
# Conversão de variável object em numeros
# Remoção de variaveis que não ajudam na previsão
# Criação de novas variáveis
# Outros tratamentos antes de criar a maquina preditiva

# Tipos de variaveis
print(dataset.dtypes)

# Removendo variavel "CPF" (não necessaria)
dataset = dataset.drop('id', axis=1)
print(dataset)

# Substituindo valores objects em numeros:
dataset['rbc'] = dataset['rbc'].replace(to_replace = {'normal': 0, 'abnormal': 1})
dataset['pc'] = dataset['pc'].replace(to_replace = {'normal': 0, 'abnormal': 1})
dataset['pcc'] = dataset['pcc'].replace(to_replace = {'present': 0, 'notpresent': 1})
dataset['ba'] = dataset['ba'].replace(to_replace = {'present': 0, 'notpresent': 1})
dataset['pcv'] = dataset['pcv'].replace(to_replace = {'\t43': 43, 'notpresent': 1})
dataset['htn'] = dataset['htn'].replace(to_replace = {'yes' : 1, 'no' : 0})
dataset['dm'] = dataset['dm'].replace(to_replace = {'\tyes':'yes', ' yes':'yes', '\tno':'no'})
dataset['dm'] = dataset['dm'].replace(to_replace = {'yes' : 1, 'no' : 0})
dataset['cad'] = dataset['cad'].replace(to_replace = {'\tno':'no'})
dataset['cad'] = dataset['cad'].replace(to_replace = {'yes' : 1, 'no' : 0})
dataset['appet'] = dataset['appet'].replace(to_replace={'good':1,'poor':0,'no':np.nan})
dataset['pe'] = dataset['pe'].replace(to_replace = {'yes' : 1, 'no' : 0})
dataset['ane'] = dataset['ane'].replace(to_replace = {'yes' : 1, 'no' : 0})
dataset['classification'] = dataset['classification'].replace(to_replace={'ckd\t':'ckd'})
dataset["classification"] = [1 if i == "ckd" else 0 for i in dataset["classification"]]
dataset['pcv'] = pd.to_numeric(dataset['pcv'], errors='coerce')
dataset['wc'] = pd.to_numeric(dataset['wc'], errors='coerce')
dataset['rc'] = pd.to_numeric(dataset['rc'], errors='coerce')

# Tratando valores nulos
print(dataset.isnull().sum().sort_values(ascending=False))

#Criando uma lista com o nome das Colunas para usar na substituição de missings)
colunas = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
           'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
           'appet', 'pe', 'ane']

# Função utilizada para varrer (loop) as colunas e a cada valor missing encontrado,
# ele será substituído pela mediana
for coluna in colunas:
    dataset[coluna] = dataset[coluna].fillna(dataset[coluna].median())

# Confere se ainda persiste valor missing
print(dataset.isnull().any().sum())
print(dataset.isnull().sum().sort_values(ascending=False))

# Verificar se existe correlação entre algumas variaveis

# Heatmap, criação da figura grafica
plt.figure(figsize=(24, 14))

# Criando um grafico heatmap
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()

# Foi verificado que as variaveis pcv e hemo possuem 85% de multicolinearidade
# Portanto estou elimando a variavel pcv
dataset.drop('pcv', axis= 1, inplace= True)

# Definindo as variaveis explicativas e o target
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Criando algoritmo de predição
# Utilizando ExtraTree para mostrar as variaveis mais importantes para o modelo
model = ExtraTreesClassifier()
model.fit(X, y)

plt.figure(figsize=(8, 6))
ranked_features = pd.Series(model.feature_importances_, index=X.columns)
ranked_features.nlargest(24).plot(kind='barh')
plt.show()

# Rankeamento das variaveis mais importantes para o modelo, neste ponto irei utilizar as 10 mais importantes
print(ranked_features.nlargest(10).index)

# Separando as 10 variaveis em uma variavel para treinamento do algoritmo
X = dataset[['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc', 'sc', 'pe']]
print(X.head()) # 5 primeiros registros
print(X.tail()) # últimos registros
print(y.head()) # Verificando o target

# Amostragem dos dados para treinametno da maquina preditiva
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

print(X_train.shape)
print(X_test.shape)

# Neste momento irei realizar as tranformações de escala referente à variáveis do meu dataset
# Criando baseline o algoritmo RandomForest

# RandomForestClassifier
# Realizando o treinamento (fit) com os dados de treino
RandomForest = RandomForestClassifier() #Criando a maquina preditiva
RandomForest = RandomForest.fit(X_train, y_train) #Treinando a maquina preditiva criada

# Realizando a previsão com dados de teste
y_pred = RandomForest.predict(X_test)

# Avaliando a performance comparando com o gabarito (y) de teste
print('Accuracy RandomForest: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Criando a MP com o algoritmo GradientBoosting
GradientBoosting = GradientBoostingClassifier(n_estimators=2000) # Criando maquina preditiva com 2000 arvores de decisões
GradientBoosting = GradientBoosting.fit(X_train, y_train) # Treinando a maquina preditiva criada

# Predictions
y_pred2 = GradientBoosting.predict(X_test)

# Performance
print('Accuracy GradientBossting: ', accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

# Exportando as maquinas preditivas
filenameRandom = 'Maquina_Preditiva_Random.pkl'
pickle.dump(RandomForest, open(filenameRandom, 'wb'))

filenameGradient = 'Maquina_Preditiva_Gradient.pkl'
pickle.dump(GradientBoosting, open(filenameGradient, 'wb'))
