# Proyecto 1

import numpy as np

import matplotlib.pyplot as plt

# Extraccion de datos del del archivo energydata_complete.csv

filename = 'energydata_complete.csv' # Nombre/ruta del archivo

# Esta funcion lee el archivo csv y guarda los datos en un array
# parametros: fname: nombre/ruta del archivo, delimiter: delimitador de las columnas, skip_header: ignora la primera linea, 
# dtype: especifica el tipo de datos (None para deteccion automatica), encoding: decodificacion (None para descartarla)
# usecols: selecciona las columnas a leer
datos = np.genfromtxt(filename, delimiter = ';', skip_header = 1, dtype = None, encoding = None, usecols = [0, 1, 2, 6])

# Se extraen los datos de humedad en un array aparte
humedad = np.array([elem[3] for elem in datos])
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

def desviacion_estandar(lista):
    return varianza(lista) ** (1/2)

def coeficiente_de_variacion(lista):
    return desviacion_estandar(lista) * 100 / promedio(lista)

def rango_muestral(lista):
    return max(lista) - min(lista)

def RIC(lista):
    cuart = cuartiles(lista)
    return cuart[2] - cuart[0]


# Cargar los datos del archivo CSV
filename = 'energydata_complete.csv'
data = np.genfromtxt(filename, delimiter=';', skip_header=1, usecols=[0, 1, 2, 6])

# Extraer los datos de humedad en un array aparte
humedad = np.array([elem[3] for elem in data])

# Generar el boxplot
fig, ax = plt.subplots()
ax.boxplot(humedad)

# Configurar las etiquetas de los ejes y el título del boxplot
ax.set_xlabel('Humedad')
ax.set_ylabel('Valores')
ax.set_title('Boxplot de Humedad')

# Mostrar el boxplot
plt.show()

# Extraccion de datos del archivo energydata_complete.csv

filename = 'energydata_complete.csv'
datos = np.genfromtxt(filename, delimiter = ';', skip_header = 1, dtype = None, encoding = None, usecols = [0, 1, 2, 6])

# Se extraen los datos de humedad en un array aparte
humedad = np.array([elem[3] for elem in datos])

# Crear un histograma de los datos de humedad
plt.hist(humedad, bins=20)

# Configurar las etiquetas de los ejes y el título del histograma
plt.xlabel('Humedad')
plt.ylabel('Frecuencia')
plt.title('Histograma de Humedad')

# Mostrar el histograma
plt.show()





print("promedio: ", promedio(humedad))
print("moda, # de repeticiones: ", moda(humedad))
print("mediana: ", mediana(humedad))
print("cuartiles [Q1, Q2, Q3]: ", cuartiles(humedad))

print("varianza: ", varianza(humedad))
print("desviacion estandar: ", desviacion_estandar(humedad))
print("coeficiente de variacion: ", coeficiente_de_variacion(humedad))
print("rango muestral: ", rango_muestral(humedad))
print("rango intercuartilico: ", RIC(humedad))
