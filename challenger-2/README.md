
# 📊 Apresentação do Grupo

**Grupo nº 76**  
**Membros:**
- Marcelo Benes Stransky Silva  
- André Dorte Cardoso  
- Wesley Mariano Mauricio  
- Gustavo Trevisan Dini

---

# 📌 Introdução do Projeto

Vamos apresentar o projeto de conclusão da segunda fase do curso, aplicando o algoritmo genético em um sistema de otimização.  
Escolhemos como solução de otimização um sistema de **alocação de recursos de portfólio de investimento de ações na bolsa de valores**.

Ou seja, a partir de algumas ações selecionadas, o algoritmo irá buscar a forma mais otimizada de distribuir percentualmente o investimento neste portfólio, levando em consideração o **risco x retorno esperado** das ações.

---

# ❓ Questionamentos

### 1. O que você está otimizando? Qual é a variável que quer maximizar ou minimizar?

Estamos optando por otimizar a **composição ideal de um portfólio de ações**, isto é, quais ativos escolher e com qual peso alocar em cada um, de forma a **maximizar o retorno esperado ajustado ao risco**.

Este é um problema de **maximização**.  
📈 Objetivo: encontrar a combinação de ativos e pesos que maximize o benefício líquido (retorno ajustado ao risco).

---

### 2. Qual é a representação da solução (genoma)?

A solução (ou indivíduo) é representada por um vetor de pesos, por exemplo:

```
[0.25, 0.15, 0.30, 0.10, 0.20]
```

- Cada valor representa a proporção investida em uma ação.
- O vetor **sempre soma 1.0** (100% do capital investido).
- O tamanho do vetor depende da quantidade de ações.

#### Método:
```python
def criar_individuo(n):
    return np.random.dirichlet(np.ones(n))
```

---

### 3. Qual é a função de fitness?

A função de fitness avalia a qualidade de um portfólio com base em **retorno esperado** e **risco (volatilidade)**.

#### Método:
```python
def calcular_fitness(pesos, mu, sigma, lambda_risco):
    retorno = np.dot(mu, pesos) 
    risco = np.sqrt(np.dot(pesos.T, np.dot(sigma, pesos)))
    return retorno - lambda_risco * risco
```

- `pesos`: vetor de alocação em cada ativo.
- `mu`: vetor de retornos médios esperados.
- `sigma`: matriz de covariância dos retornos.
- `lambda_risco`: grau de aversão ao risco do investidor.

#### Interpretação:
- **Retorno**: ganho potencial do portfólio.
- **Risco**: volatilidade do portfólio.
- **Fitness final**: `Retorno - λ * Risco`

🧠 Esta fórmula segue o conceito de utilidade ou índice de Sharpe simplificado.

---

### 4. Qual é o método de seleção?

Usamos **Seleção por Torneio (Tournament Selection)**.

#### Como funciona:
- Seleciona aleatoriamente `k` indivíduos da população (ex: `k=3`).
- Escolhe o de maior fitness como pai.

#### Vantagens:
- Simples, eficiente.
- Favorece bons indivíduos mantendo diversidade.

#### Método:
```python
def selecionar_pais(populacao, fitness, k=3):
    pais = random.choices(list(zip(populacao, fitness)), k=k)
    return max(pais, key=lambda x: x[1])[0]
```

---

### 5. Qual é o método de crossover?

Usamos o **crossover aritmético** para combinar pais.

#### Como funciona:
- Gera `α ∈ [0,1]` aleatório
- Combina: `filho = α ⋅ pai1 + (1−α) ⋅ pai2`
- Normaliza para soma = 1

#### Método:
```python
def crossover(pai1, pai2):
    alpha = np.random.rand()
    filho = alpha * pai1 + (1 - alpha) * pai2
    return filho / np.sum(filho)
```

---

### 6. Qual será o método de inicialização?

A população inicial é criada com a **distribuição de Dirichlet**, que gera vetores com soma = 1.

#### Métodos:
```python
def criar_individuo(n):
    return np.random.dirichlet(np.ones(n))

def criar_populacao(tamanho, n_acoes):
    return [criar_individuo(n_acoes) for _ in range(tamanho)]
```

#### Vantagens:
- Normalização garantida
- Boa diversidade inicial
- Eficiência matemática

---

### 7. Qual o critério de parada?

- **Máximo de gerações** (ex: 200 gerações)
- **Estagnação**: para se o melhor fitness não melhora em X gerações consecutivas (ex: 50)

🛑 Isso evita execuções desnecessárias quando a solução já convergiu.

---

# ⚙️ Dependências do Projeto

- `streamlit`  
- `numpy`  
- `pandas`  
- `yfinance`  
- `matplotlib`  
- `random`

---

# ▶️ Comando Para Execução da Interface

```bash
streamlit run portifolio_investimento.py
```
