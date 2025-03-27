import numpy as np
from sklearn.metrics import mean_squared_error
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


area = np.array([32, 149, 78, 132, 155, 178, 201, 224, 247, 220])
custo = np.array([51000, 265000, 110000, 201000, 230500, 260000, 289500, 319000, 348500, 315000])


def calcular_erro(w, b, x, y):
    y_pred = w * x + b
    mse = mean_squared_error(y, y_pred)
    return mse


w_values = np.linspace(0, 2000, 20)  # Valores de w (coeficiente angular)
b_values = np.linspace(0, 500000, 20)  # Valores de b (coeficiente linear)


melhor_w = 0
melhor_b = 0
menor_erro = float('inf')


for w in w_values:
    for b in b_values:
        erro = calcular_erro(w, b, area, custo)
        if erro < menor_erro:
            menor_erro = erro
            melhor_w = w
            melhor_b = b


def gerar_pdf():
    c = canvas.Canvas("relatorio.pdf", pagesize=letter)
    c.drawString(100, 750, "Relatório de Experimentos com Regressão Linear")

    c.drawString(100, 730, f"Melhor w: {melhor_w}")
    c.drawString(100, 710, f"Melhor b: {melhor_b}")
    c.drawString(100, 690, f"Menor erro (MSE): {menor_erro}")
    
    c.drawString(100, 670, "Resultados dos experimentos de w e b:")


    y_position = 650
    for w in w_values:
        for b in b_values:
            c.drawString(100, y_position, f"w: {w}, b: {b}")
            y_position -= 10

  
    c.drawString(100, 200, "Código Python utilizado:")
    code = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

area = np.array([32, 149, 78, 132, 155, 178, 201, 224, 247, 220])
custo = np.array([51000, 265000, 110000, 201000, 230500, 260000, 289500, 319000, 348500, 315000])

def calcular_erro(w, b, x, y):
    y_pred = w * x + b
    mse = mean_squared_error(y, y_pred)
    return mse

w_values = np.linspace(0, 2000, 20)
b_values = np.linspace(0, 500000, 20)

melhor_w = 0
melhor_b = 0
menor_erro = float('inf')

for w in w_values:
    for b in b_values:
        erro = calcular_erro(w, b, area, custo)
        if erro < menor_erro:
            menor_erro = erro
            melhor_w = w
            melhor_b = b

print(f"Melhor w: {melhor_w}")
print(f"Melhor b: {melhor_b}")
print(f"Menor erro (MSE): {menor_erro}")

plt.scatter(area, custo, color='blue', label='Dados reais')
plt.plot(area, melhor_w * area + melhor_b, color='red', label='Melhor modelo')
plt.xlabel('Área (m²)')
plt.ylabel('Custo (R$)')
plt.legend()
plt.show()
    """
    c.drawString(100, 180, code)

    c.save()

gerar_pdf()

print(f"Melhor w: {melhor_w}")
print(f"Melhor b: {melhor_b}")
print(f"Menor erro (MSE): {menor_erro}")
