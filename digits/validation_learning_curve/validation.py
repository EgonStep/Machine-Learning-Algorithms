import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import validation_curve, learning_curve

db = datasets.load_digits()

X = db.data
Y = db.target

np.random.seed(0)

n_samples = len(X)
percentage = 0.75

order = np.random.permutation(n_samples)

X = X[order]
Y = Y[order]

Y_teste = Y[int(percentage*n_samples):]
X_teste = X[int(percentage*n_samples):]

Y_treino = Y[:int(percentage*n_samples)]
X_treino = X[:int(percentage*n_samples)]

C = [x for x in range(1, 11)]

clf = svm.SVC(gamma='scale')

# No validation curse o atributo "cv=5" vai treinar 1 e validar  4, separando em grupos de 5 em 5
train_scores, validation_scores = validation_curve(clf, X_treino, Y_treino, "C", C, verbose=3, cv=5)
train_sizes, train_scores_lc, validation_scores_lc = learning_curve(clf, X_treino, Y_treino, cv=5)

# Verificar se o desvio padrao não se distancia muito da media
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

train_scores_mean_lc = np.mean(train_scores_lc, axis=1)
train_scores_std_lc = np.std(train_scores_lc, axis=1)
validation_scores_mean_lc = np.mean(validation_scores_lc, axis=1)
validation_scores_std_lc = np.std(validation_scores_lc, axis=1)

plt.subplot(1, 2, 1)
plt.title("Curva de validação com SVM")
plt.xlabel("C")
plt.ylabel("Score")

min_all = np.min([np.min(train_scores), np.min(validation_scores)])

plt.ylim(min_all, 1.001)
plt.grid()
param_range = list(range(1, len(C)+1))
# Largura da linha
lw = 2

# Plotagem do grafico
plt.plot(param_range, train_scores_mean, 'o-', label='Training Score', lw=lw)
plt.fill_between(param_range, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.2, lw=lw)

plt.plot(param_range, validation_scores_mean, 'o-', label='Validation Score', lw=lw)
plt.fill_between(param_range, validation_scores_mean-validation_scores_std,
                 validation_scores_mean+validation_scores_std, alpha=0.2, lw=lw)

plt.legend(loc="best")

plt.subplot(1,2,2)

plt.title("Curva de aprendizado com SVM")
plt.xlabel("Exemplos de Treino")
plt.ylabel("Score")
plt.grid()

min_all = np.min([np.min(train_scores_lc),np.min(validation_scores_lc)])

plt.ylim(min_all,1.001)

# Largura da linha
lw = 2
plt.plot(train_sizes,train_scores_mean_lc,'o-',label='Training Score',lw=lw)
plt.fill_between(train_sizes,train_scores_mean_lc-train_scores_std_lc,train_scores_mean_lc+train_scores_std_lc,alpha=0.2,lw=lw)

plt.plot(train_sizes,validation_scores_mean_lc,'o-',label='Validation Score',lw=lw)
plt.fill_between(train_sizes,validation_scores_mean_lc-validation_scores_std_lc,validation_scores_mean_lc+validation_scores_std_lc,alpha=0.2,lw=lw)

plt.legend(loc="best")

plt.show()

'''
- Underfit modelo nao consegue especificar as informações muito bem, trata todo os dados como algo parecido 
    (treino e validação estao baixos)
- Overfiting se especializou demais, o algoritmo é muito bom para saber sobre a base de treino mas quando entra 
    valores novos podera apresentará dados não muito confiaveis (treino alto e validação baixo)
- Ideia que o algoritmo fique proximo do ponto maximo da base de treino (grafico curva de validação)
- Curva de aprendizado verifica se a base de dados auxilia na melhora do algoritmo (tendencia de como o 
    classificador se comporta com mais elementos)
'''
