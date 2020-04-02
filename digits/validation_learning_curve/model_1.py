import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold


db = datasets.load_digits()
X = db.data
Y = db.target

'''
- Elimina os atributos com variancia baixa ou iguais.
- Exemplo banana e maca tem um atributo = 0, ou seja, 
    nao podemos extrair nenhuma informação desse atributo, pois são iguais
    em ambos os elementos.
- (Threshold) Se os atributos de dados foram 90% iguais, serão eliminados da analise.
- Normalmente é melhor usar o threshold padrao
'''

sel = VarianceThreshold(threshold=(0.9*(1-0.9)))

X_new = sel.fit_transform(X)

print(len(X[0]))
print(len(X_new[0]))

variancia = sel.variances_
features = list(range(1, len(X[0])+1))

plt.title("Variância sobre os atributos de Digitos.")
plt.xlabel("Atributo")
plt.ylabel("Variância")

plt.bar(features,variancia)

for index, data in enumerate(features):
    plt.text(x=index+0.6,y=variancia[index]+0.5,s='{0:.1f}'.format(variancia[index]),fontdict=dict(fontsize=8),rotation=90)

plt.show()