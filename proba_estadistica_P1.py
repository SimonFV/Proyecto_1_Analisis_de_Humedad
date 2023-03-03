"""
Probabilidad y estadística
Grupo 3

Integrantes:
Simón Josue Fallas Villalobos 
Irene Patricia Muñoz Castro 
Luis Felipe Vargas Jiménez 

Proyecto 1 - Estadística Descriptiva

Medidas de tendencia y dispersión de un conjunto de medidas de % de humedad del 
ambiente en la sala por medio de sensores, junto con diferentes representaciones
gráficas de estos datos.

"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Extraccion de datos del del archivo energydata_complete.csv

filename = 'energydata_complete.csv' # Nombre/ruta del archivo

# Esta funcion lee el archivo csv y guarda los datos en un array
# parametros: fname: nombre/ruta del archivo, delimiter: delimitador de las columnas, skip_header: ignora la primera linea, 
# dtype: especifica el tipo de datos (None para deteccion automatica), encoding: decodificacion (None para descartarla)
# usecols: selecciona las columnas a leer
datos = np.genfromtxt(filename, delimiter = ';', skip_header = 1, dtype = None, encoding = None, usecols = [0, 6])

# Se extraen los datos de humedad y tiempo en arrays aparte
_tiempo = np.array([elem[0] for elem in datos])
tiempo = [dt.datetime.strptime(d,'%d/%m/%Y %H:%M') for d in _tiempo]
humedad_sin_orden = np.array([elem[1] for elem in datos])
humedad = np.array(humedad_sin_orden, copy=True)

# Se ordenan los datos usando la funcion sort de numpy, utilizando el algoritmo quicksort
humedad.sort(kind = 'quicksort')

# Funcion que calcula el promedio de un array
def promedio(lista):
    total = 0
    for n in range(len(lista)):
        total += lista[n]
    return total/(n + 1)

# Funcion que calcula la moda de un array
def moda(lista):
    # Se organiza el array en un diccionario y se cuenta el numero de repeticiones de cada valor
    valores = dict()
    for valor in lista:
        if valor in valores.keys():
            valores[valor] += 1
        else:
            valores[valor] = 1
    # Se buscan los valores maximos del diccionario, cuyas llaves seran la moda
    maximo = 1
    moda = []
    for key in valores:
        if valores[key] == maximo:
            moda.append([key, valores[key]])
        elif valores[key] > maximo:
            moda = [[key, valores[key]]]
            maximo = valores[key]
    return moda

# Funcion que calcula la mediana de un array
def mediana(lista):
    n = len(lista)
    if n % 2 == 0: 
        return (lista[int(n / 2) - 1] + lista[int((n / 2))]) / 2 # n par
    return lista[int((n + 1) / 2) - 1] # n impar

# Funcion que calcula los cuartiles de un array
def cuartiles(lista):
    n = len(lista)
    if n % 4 == 0:
        return [lista[int(n / 4) - 1], mediana(lista), lista[int(n * 3 / 4) - 1]] # n divisible por 4
    return [lista[int(n / 4)], mediana(lista), lista[int(n * 3 / 4)]] # n no divisible por 4

# Funcion que calcula la varianza de un array
def varianza(lista):
    prom = promedio(lista)
    varianza = 0
    for n in range(len(lista)):
        varianza += (lista[n] - prom) ** 2
    return varianza / n

# Funcion que calcula la desviasion estandar de un array
def desviacion_estandar(lista):
    return varianza(lista) ** (1/2)

# Funcion que calcula el coeficiente de variacion un array (porcentual)
def coeficiente_de_variacion(lista):
    return desviacion_estandar(lista) * 100 / promedio(lista)

# Funcion que calcula el rango muestral de un array
def rango_muestral(lista):
    return max(lista) - min(lista)
    
# Funcion que calcula el rango intercuartil de un array
def RIC(lista):
    cuart = cuartiles(lista)
    return cuart[2] - cuart[0]

_promedio = promedio(humedad)
_moda = moda(humedad)
_mediana = mediana(humedad)
_cuartiles = cuartiles(humedad)
_varianza = varianza(humedad)
_desviacion_estandar = desviacion_estandar(humedad)
_coeficiente_de_variacion = coeficiente_de_variacion(humedad)
_rango_muestral = rango_muestral(humedad)
_RIC = RIC(humedad)

print('---MEDIDAS DE TENDENCIA CENTRAL---')
print("Promedio: ", _promedio)
print("Moda, # de repeticiones: ", _moda)
print("Mediana: ", _mediana)
print("Cuartiles [Q1, Q2, Q3]: ", _cuartiles, '\n')

print('---MEDIDAS DE VARIABILIDAD O DISPERSIÓN---')
print("Varianza: ", _varianza)
print("Desviacion estandar: ", _desviacion_estandar)
print("Coeficiente de variacion: ", _coeficiente_de_variacion)
print("Rango muestral: ", _rango_muestral)
print("Rango intercuartilico: ", _RIC)

#-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Diagramas  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-
'''(comentar para generar gráfico lineal [agrega#])    
#----------------------Generar el plot de lineal--------------------------------

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.plot(tiempo, humedad_sin_orden)
plt.gcf().autofmt_xdate()

plt.xlabel('Tiempo')
plt.ylabel('%Humedad')
plt.title("Gráfico lineal de %humedad en función del tiempo en el año 2016")
plt.show()
#'''

'''(comentar para generar graf lineal zoom [agrega#])
# -------------------Generar el plot de grafico lineal zoom------------------------------

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.    gca().xaxis.set_major_locator(mdates.HourLocator(interval=8))
i, j = tiempo.index(dt.datetime(2016, 5, 2, 0, 0, 0)), tiempo.index(dt.datetime(2016, 5, 5, 0, 0, 0))
plt.plot(tiempo[i:j], humedad_sin_orden[i:j])
plt.gcf().autofmt_xdate()

plt.xlabel('Tiempo')
plt.ylabel('%Humedad')
plt.title("Gráfico lineal de %humedad en función del tiempo en el año 2016")
plt.show()
#'''

'''(comentar para generar digrama de caja [agrega#])
# -------------------------Generar el boxplot----------------------------------
fig, ax = plt.subplots()
ax.boxplot(humedad)

# Configurar las etiquetas de los ejes y el título del boxplot
ax.set_xlabel('Humedad')
ax.set_ylabel('Valores')
ax.set_title('Boxplot de %Humedad')

# Mostrar el boxplot
plt.show()
'''

# Numero de celdas
n = round(np.sqrt(len(humedad)))

# Calcular el ancho del intervalo
ancho = ((_rango_muestral) * 0.05 + _rango_muestral) / n

# Crear la figura del histograma
plt.figure(figsize=(20, 10))

# Crear el histograma con 142 celdas, con una anchura de 0.1
plt.hist(humedad, bins = n, range = (humedad[0], humedad[len(humedad) - 1]), width = ancho, rwidth=0.5)

# Configurar las etiquetas de los ejes y el título del histograma
plt.xlabel('% Humedad')
plt.ylabel('Frecuencia')
plt.title('Histograma de % de Humedad')

# Aca se muestra el histograma
plt.show()
