import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import random


# ========================
# Funções do Algoritmo
# ========================

def obter_retorno_e_cov(acoes, periodo="1y"):
    dados = yf.download(acoes, period=periodo, auto_adjust=True)['Close']
    retornos = np.log(dados / dados.shift(1)).dropna()
    retorno_esperado = retornos.mean()
    covariancia = retornos.cov()
    return retorno_esperado, covariancia


def calcular_fitness(pesos, mu, sigma, lambda_risco):
    retorno = np.dot(mu, pesos)
    risco = np.sqrt(np.dot(pesos.T, np.dot(sigma, pesos)))
    return retorno - lambda_risco * risco


def criar_individuo(n):
    return np.random.dirichlet(np.ones(n))


def criar_populacao(tamanho, n_acoes):
    return [criar_individuo(n_acoes) for _ in range(tamanho)]


def selecionar_pais(populacao, fitness, k=3):
    pais = random.choices(list(zip(populacao, fitness)), k=k)
    return max(pais, key=lambda x: x[1])[0]


def crossover(pai1, pai2):
    alpha = np.random.rand()
    filho = alpha * pai1 + (1 - alpha) * pai2
    return filho / np.sum(filho)  # Normaliza para manter soma 1


def mutar(individuo, taxa=0.1):
    # taxa=0.1 → muta 10% dos indivíduos em média por geração.
    # Se for muito baixa, o algoritmo pode ficar preso em ótimos locais.
    # Se for muito alta, o algoritmo pode perder boas soluções e virar algo quase aleatório.
    if np.random.rand() < taxa:
        perturbacao = np.random.normal(0, 0.02, size=len(individuo))
        individuo = individuo + perturbacao
        individuo = np.clip(individuo, 0, 1)
        individuo /= np.sum(individuo)
    return individuo


def algoritmo_genetico(mu, sigma, lambda_risco, geracoes=200, pop_tam=50, taxa_mutacao=0.1, elitismo=0.1):
    melhores_fitness = []
    n = len(mu)
    populacao = criar_populacao(pop_tam, n)
    melhor_fitness_historico = -np.inf
    geracoes_sem_melhora = 0
    limite_estagnacao = 50  # Novo parâmetro de critério de parada

    for gen in range(geracoes):
        fitness = [calcular_fitness(ind, mu, sigma, lambda_risco) for ind in populacao]
        melhores_fitness.append(max(fitness))

        fitness_max = max(fitness)
        if fitness_max > melhor_fitness_historico:
            melhor_fitness_historico = fitness_max
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        # Parar por estagnação
        if geracoes_sem_melhora >= limite_estagnacao:
            print(f"Parando por estagnação após {gen + 1} gerações.")
            break

        num_elite = int(elitismo * pop_tam)
        elite = [x for _, x in sorted(zip(fitness, populacao), key=lambda x: x[0], reverse=True)][:num_elite]
        nova_populacao = elite.copy()

        while len(nova_populacao) < pop_tam:
            pai1 = selecionar_pais(populacao, fitness)
            pai2 = selecionar_pais(populacao, fitness)
            filho = crossover(pai1, pai2)
            filho = mutar(filho, taxa_mutacao)
            nova_populacao.append(filho)

        populacao = nova_populacao

    fitness_final = [calcular_fitness(ind, mu, sigma, lambda_risco) for ind in populacao]
    melhor_indice = np.argmax(fitness_final)
    melhor_solucao = populacao[melhor_indice]
    melhor_retorno = np.dot(mu, melhor_solucao)
    melhor_risco = np.sqrt(np.dot(melhor_solucao.T, np.dot(sigma, melhor_solucao)))

    return melhor_solucao, melhor_retorno, melhor_risco, melhores_fitness


# ========================
# Interface Streamlit
# ========================

st.title("🧬 Otimização de Portfólio com Algoritmo Genético.")

lista_acoes = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'WEGE3.SA', 'ABEV3.SA', 'BBDC4.SA', 'JBSS3.SA']
acoes_selecionadas = st.multiselect("Selecione as ações para o portfólio:", lista_acoes,
                                    default=['ABEV3.SA', 'BBDC4.SA'])

lambda_risco = st.slider("Aversão ao risco (λ)", 0.0, 2.0, 0.5, step=0.1)
geracoes = st.slider("Nº de gerações", 10, 500, 200, step=10)
pop_tam = st.slider("Tamanho da população", 10, 200, 50, step=10)
taxa_mutacao = st.slider("Taxa de mutação", 0.0, 1.0, 0.1, step=0.05)
elitismo = st.slider("Porcentagem de elitismo", 0.0, 1.0, 0.1, step=0.05)

if st.button("Rodar Otimização"):
    with st.spinner("🧠 Rodando algoritmo genético..."):
        mu, sigma = obter_retorno_e_cov(acoes_selecionadas)
        solucao, retorno, risco, historico = algoritmo_genetico(
            mu, sigma, lambda_risco, geracoes, pop_tam, taxa_mutacao, elitismo
        )

    st.success("🏁 Otimização concluída!")

    st.subheader("📊 Portfólio Ótimo:")
    resultado = pd.DataFrame({
        "Ação": acoes_selecionadas,
        "Peso (%)": np.round(solucao * 100, 2)
    })
    st.dataframe(resultado)

    st.markdown(f"**📈 Retorno Esperado:** {retorno:.2%}")
    st.markdown(f"**📉 Risco (Volatilidade):** {risco:.2%}")
    st.markdown(f"**🏆 Fitness Final:** {retorno - lambda_risco * risco:.5f}")

    st.subheader("📈 Evolução do Fitness")
    fig, ax = plt.subplots()
    ax.plot(historico)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    import pandas as pd

    # Certifica que o índice 'Geração' é inteiro
    df_historico = pd.DataFrame({
        "Geração": list(range(len(historico))),  # Garante números inteiros
        "Fitness": historico  # Pode conter valores negativos
    }).astype({"Geração": int}).set_index("Geração")

    st.subheader("📈 Evolução do Fitness")
    st.line_chart(df_historico)


