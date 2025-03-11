import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_fixed_bet_portfolio(initial_cap, base_win_prob, win_prob_std, num_trades, bet_fraction, gain, random_seed=None):
    """
    Simula la evolución del portafolio apostando una fracción fija del portafolio en cada apuesta.
    La probabilidad de ganar se genera para cada apuesta a partir de una distribución normal centrada en base_win_prob
    con desviación estándar win_prob_std, truncada entre 0 y 1.
    
    Parámetros:
      initial_cap: capital inicial.
      base_win_prob: probabilidad base de ganar cada apuesta.
      win_prob_std: desviación estándar de la probabilidad.
      num_trades: número total de apuestas.
      bet_fraction: fracción del portafolio apostada en cada jugada.
      gain: ganancia decimal en caso de victoria.
      random_seed: semilla para reproducibilidad.
    
    Retorna:
      portfolio: Array con la evolución del portafolio.
      win_probabilities: Array con la probabilidad de ganar usada en cada apuesta.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    portfolio = np.zeros(num_trades + 1)
    portfolio[0] = initial_cap
    win_probabilities = np.zeros(num_trades)
    
    for i in range(1, num_trades+1):
        # Probabilidad de ganar para la apuesta actual (truncada entre 0 y 1)
        current_win_prob = np.clip(np.random.normal(loc=base_win_prob, scale=win_prob_std), 0, 1)
        win_probabilities[i-1] = current_win_prob
        
        # Cálculo del monto a apostar: una fracción del portafolio actual
        bet = bet_fraction * portfolio[i-1]
        if np.random.rand() < current_win_prob:
            # Si gana: se suma la ganancia
            portfolio[i] = portfolio[i-1] + gain * bet
        else:
            # Si pierde: se descuenta el monto apostado
            portfolio[i] = portfolio[i-1] - bet
    return portfolio, win_probabilities

# Título de la aplicación
st.title("Simulación de Portafolio usando Kelly Criterion")

st.write("""
Esta aplicación simula una serie de apuestas donde se arriesga una fracción fija del portafolio en cada jugada.
Para cada apuesta, se genera una probabilidad de ganar a partir de una distribución normal centrada en una probabilidad base,
con cierta variabilidad. Si se gana, se suma una ganancia proporcional al monto apostado; si se pierde, se descuenta ese monto.
Puedes ajustar los parámetros de la simulación desde la barra lateral y visualizar cómo evoluciona el portafolio y las probabilidades.
""")

st.markdown("**por Fernando Guzman**")

# Parámetros de simulación en la barra lateral
st.sidebar.header("Parámetros de Simulación")
initial_cap = st.sidebar.number_input("Capital inicial", value=7000)
base_win_prob = st.sidebar.number_input("Probabilidad base de ganar", value=0.60, min_value=0.0, max_value=1.0)
win_prob_std = st.sidebar.number_input("Desviación estándar de la probabilidad", value=0.05, min_value=0.0)
num_trades = st.sidebar.number_input("Número total de apuestas", value=50, min_value=1)
bet_fraction = st.sidebar.number_input("Fracción del portafolio apostada", value=0.02, min_value=0.0, max_value=1.0)
gain = st.sidebar.number_input("Ganancia en caso de victoria", value=0.67, min_value=0.0)
seed = st.sidebar.number_input("Semilla", value=42, step=1)

if st.sidebar.button("Simular"):
    # Ejecuta la simulación
    portfolio, win_probabilities = simulate_fixed_bet_portfolio(
        initial_cap, base_win_prob, win_prob_std, num_trades, bet_fraction, gain, random_seed=seed
    )
    
    # Cálculo del umbral de apuesta positiva
    bet_multiplier_if_win = 1 + gain * bet_fraction
    bet_multiplier_if_loss = 1 - bet_fraction
    win_prob_threshold = (1 - bet_multiplier_if_loss) / (bet_multiplier_if_win - bet_multiplier_if_loss)
    
    st.write("Umbral de probabilidad para apuesta positiva: **{:.2f}**".format(win_prob_threshold))
    optimal_decision = "Apostar" if base_win_prob > win_prob_threshold else "No apostar"
    st.write("Con base_win_prob = **{:.2f}**, la estrategia es: **{}**".format(base_win_prob, optimal_decision))
    
    st.write("### Evolución del portafolio")
    st.write(portfolio)
    
    # Gráfica de la evolución del portafolio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(portfolio)), portfolio, 'r-', label='Portafolio')
    ax.set_title("Evolución del Portafolio")
    ax.set_xlabel("Número de Apuestas")
    ax.set_ylabel("Valor del Portafolio")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    st.write("### Probabilidad de Ganar en Cada Apuesta")
    # Gráfica de la probabilidad de ganar en cada apuesta
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(range(1, num_trades+1), win_probabilities, 'bo-')
    ax2.set_title("Probabilidad de Ganar en Cada Apuesta")
    ax2.set_xlabel("Apuesta")
    ax2.set_ylabel("Probabilidad")
    ax2.grid(True)
    st.pyplot(fig2)
    
    st.write("Valor final del portafolio: **{:.2f}**".format(portfolio[-1]))
