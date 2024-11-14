import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Función para generar el archivo de productos correspondientes al padrón
def generar_archivo_padron(padron, parts_per_padron_path, output_path):
    # Cargar el archivo CSV
    parts_per_padron = pd.read_csv(parts_per_padron_path)
    
    # Extraer los dos últimos dígitos del padrón
    padron_suffix = str(padron)[-2:]
    columna_padron = f'ends {padron_suffix}'

    # Filtrar los productos correspondientes
    productos_vinculados = parts_per_padron[parts_per_padron[columna_padron] == 1]

    # Crear un nuevo DataFrame con el formato de sampleSubmission
    resultado = pd.DataFrame({
        'id': productos_vinculados['id'],
        'ROP': 0,  # Inicialmente en cero como se indicó
        'Q': 0     # Inicialmente en cero como se indicó
    })

    # Guardar el resultado en un nuevo archivo CSV
    resultado.to_csv(output_path, index=False)
    return productos_vinculados['id'].tolist()

# Función para proyectar la demanda de los próximos 100 días
def proyectar_demanda(part_demand_path, productos_ids, output_path):
    # Cargar el archivo de demanda
    part_demand = pd.read_csv(part_demand_path)
    
    # Preparar el DataFrame para la proyección
    proyeccion = pd.DataFrame(columns=['id'] + [f'Day {i+1}' for i in range(100)])
    
    for producto_id in productos_ids:
        # Filtrar los datos del producto actual
        demanda_producto = part_demand[part_demand['id'] == producto_id].drop(columns='id').values.flatten()
        
        # Ajustar la regresión lineal
        X = np.array(range(len(demanda_producto))).reshape(-1, 1)
        y = demanda_producto
        reg = LinearRegression().fit(X, y)
        
        # Proyectar los próximos 100 días
        X_futuro = np.array(range(len(demanda_producto), len(demanda_producto) + 100)).reshape(-1, 1)
        y_futuro = reg.predict(X_futuro)
        
        # Calcular la desviación estándar
        desviacion = y.std()
        
        # Generar la demanda proyectada para los próximos 100 días
        proyeccion_producto = {
            'id': producto_id,
            **{f'Day {i+1}': int(np.round(np.random.normal(loc=media, scale=desviacion))) for i, media in enumerate(y_futuro)}
        }
        
        # Añadir la proyección al DataFrame
        proyeccion = proyeccion.append(proyeccion_producto, ignore_index=True)
    
    # Guardar el archivo de proyección
    proyeccion.to_csv(output_path, index=False)

# Función para calcular Q y ROP día a día
def calcular_q_rop_diario(part_data, demanda_proyectada):
    # Convertir costos anuales a diarios
    c1_diario = part_data['c1'] / 300
    c2_diario = part_data['c2'] / 300
    k = part_data['k']
    lt = part_data['lt']
    
    # Inicializar listas para guardar Q y ROP diarios
    q_diario = []
    rop_diario = []
    
    for dia in range(100):
        D = demanda_proyectada[f'Day {dia+1}'].mean()
        std_dev = demanda_proyectada[f'Day {dia+1}'].std()

        # Considerar la demanda negativa como cero
        D = max(D, 0)
        
        # Verificar que D y c1_diario sean mayores que cero para evitar valores inválidos
        if D > 0 and c1_diario > 0:
            Q_optimo = np.sqrt((2 * k * D) / (c1_diario * (1 + (c1_diario / c2_diario))))
            S_optimo = Q_optimo * (c2_diario / (c1_diario + c2_diario))
            ROP = (lt * D) - (Q_optimo - S_optimo)
            
            q_diario.append(int(np.round(Q_optimo)))
            rop_diario.append(int(np.round(ROP)))
        else:
            q_diario.append(0)
            rop_diario.append(0)
    
    return q_diario, rop_diario

# Simular la operación del almacén durante 100 días
def simular_inventario(part_data, demanda_proyectada, q_diario, rop_diario, output_path):
    # Inicializar variables
    stock_inicial = part_data['initialStock']
    volumen_unitario = part_data['unitVolume']
    capacidad_maxima = stock_inicial * volumen_unitario
    costo_total = 0
    
    # Inicializar DataFrame para los resultados
    resultados = pd.DataFrame(columns=['Day', 'Stock', 'ROP', 'Q', 'Pedido', 'Costo'])
    
    for dia in range(100):
        demanda_dia = demanda_proyectada[f'Day {dia+1}'].iloc[0]
        stock = max(stock_inicial - demanda_dia, 0)
        pedido = 0
        costo = 0
        
        # Verificar si se necesita hacer un pedido
        if stock < rop_diario[dia]:
            pedido = q_diario[dia]
            stock += pedido
        
        # Calcular el costo del día
        costo += stock * part_data['c1'] / 300
        if stock < 0:
            costo += abs(stock) * part_data['c2'] / 300
        
        # Verificar si se supera la capacidad máxima
        if stock * volumen_unitario > capacidad_maxima:
            costo *= 2
        
        # Guardar los resultados del día
        resultados = resultados.append({
            'Day': dia+1,
            'Stock': stock,
            'ROP': rop_diario[dia],
            'Q': q_diario[dia],
            'Pedido': pedido,
            'Costo': costo
        }, ignore_index=True)
        
        # Actualizar el stock inicial para el siguiente día
        stock_inicial = stock
    
    # Guardar los resultados en un archivo CSV
    resultados.to_csv(output_path, index=False)
    
    # Calcular el costo total esperado diario promedio
    cte_diario_promedio = resultados['Costo'].mean()
    
    return cte_diario_promedio

# Rutas de los archivos
parts_per_padron_path = r'C:\Users\user\Documents\data\partsPerPadron.csv'
part_data_path = r'C:\Users\user\Documents\data\partData.csv'
part_demand_path = r'C:\Users\user\Documents\data\partDemand.csv'
output_padron_path = r'C:\Users\user\Documents\data\poli_96577.csv'
output_proyeccion_path = r'C:\Users\user\Documents\data\demanda_proyectada.csv'
output_simulacion_path = r'C:\Users\user\Documents\data\simulacion_inventario.csv'

# Padrón a analizar
padron = 96577

# Generar el archivo de productos correspondientes al padrón
productos_ids = generar_archivo_padron(padron, parts_per_padron_path, output_padron_path)

# Proyectar la demanda de los próximos 100 días
proyectar_demanda(part_demand_path, productos_ids, output_proyeccion_path)

# Cargar los datos de partData.csv
part_data = pd.read_csv(part_data_path)

# Leer la demanda proyectada
demanda_proyectada = pd.read_csv(output_proyeccion_path)

# Calcular ROP y Q para cada producto y simular el inventario
cte_diario_promedio_total = 0
promedios_q = []
promedios_rop = []

for producto_id in productos_ids:
    datos_producto = part_data[part_data['id'] == producto_id].iloc[0]
    demanda_proyectada_producto = demanda_proyectada[demanda_proyectada['id'] == producto_id]
    
    q_diario, rop_diario = calcular_q_rop_diario(datos_producto, demanda_proyectada_producto)
    
    # Calcular los promedios de Q y ROP
    promedio_q = int(np.round(np.mean(q_diario)))
    promedio_rop = int(np.round(np.mean(rop_diario)))
    promedios_q.append(promedio_q)
    promedios_rop.append(promedio_rop)
    
    cte_diario_promedio = simular_inventario(datos_producto, demanda_proyectada_producto, q_diario, rop_diario, output_simulacion_path)
    cte_diario_promedio_total += cte_diario_promedio

# Calcular el costo total esperado diario promedio para todos los productos
cte_diario_promedio_total /= len(productos_ids)
print(f'Costo total esperado diario promedio: {cte_diario_promedio_total}')

# Actualizar el archivo poli_96577.csv con los promedios de Q y ROP
poli_96577 = pd.read_csv(output_padron_path)
poli_96577['Q'] = promedios_q
poli_96577['ROP'] = promedios_rop
poli_96577.to_csv(output_padron_path, index=False)

