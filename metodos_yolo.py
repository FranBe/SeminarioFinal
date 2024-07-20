import subprocess
# Clases
nombres_clases = ['spurious_copper', 'mouse_bite', 'open_circuit', 'missing_hole', 'spur', 'short']
# Cantidad de clases
numero_clases = 6

def YOLO_ajusta_yaml(ruta_archivo_yaml, ruta_entrenamiento, ruta_validacion, numero_clases, nombres_clases):
    """
    Crea un archivo YAML con el contenido especificado para configurar los datos de entrenamiento y validación.

    Parámetros:
    ruta_archivo_yaml (str): Ruta donde se guardará el archivo YAML.
    ruta_entrenamiento (str): Ruta a la carpeta de entrenamiento.
    ruta_validacion (str): Ruta a la carpeta de validación.
    numero_clases (int): Número de clases.
    nombres_clases (list): Lista de nombres de las clases.
    """
    # Modifica el archivo YAML, también puede hacerse a mano
    data_yaml_content = f"""
    train: {ruta_entrenamiento}
    val: {ruta_validacion}
    nc: {numero_clases}
    names: {nombres_clases}
    """
    # Crear y escribir el archivo YAML en la ruta especificada
    with open(ruta_archivo_yaml, 'w') as f:
        f.write(data_yaml_content)

def YOLO_entrenar_modelo():
    # Hiperparámetros del entrenamiento
    comando = [
        'python', 'train.py',
        '--img-size', '640',
        '--batch-size', '16',
        '--epochs', '100',
        '--data', 'data.yaml',
        '--cfg', 'models/yolov5s.yaml',
        '--weights', 'yolov5s.pt',
        '--name', 'my_experiment',
        '--save-period', '1',
        '--project', 'runs/'
    ]

    # Ejecutar el comando
    result = subprocess.run(comando, capture_output=True, text=True)

    # Salida
    print("Salida del comando:")
    print(result.stdout)

    # Si hay error
    if result.stderr:
        print("Errores del comando:")
        print(result.stderr)

def YOLO_test_de_imagen():
    comando = [
        '/detect.py',
        '--weights', '/yolov5/runs/my_experiment6/weights/best.pt',
        '--img-size', '640',
        '--conf', '0.5',
        '--source', '/imagenes_para_probar',
        '--save-txt',
        '--save-conf',
        '--project', '/yolov5/runs/detect/'
    ]

    # Ejecutar el comando
    result = subprocess.run(comando, capture_output=True, text=True)
    # Salida
    print("Salida del comando:")
    print(result.stdout)
    # Si hay error
    if result.stderr:
        print("Errores del comando:")
        print(result.stderr)
        

######################################################################################
#Iniciar el YOLO, antes setear correctamente los parámetros
######################################################################################
ruta_archivo_yaml = '/yolov5/data.yaml'
ruta_entrenamiento = '/PCB_split/train'
ruta_validacion = '/PCB_split/val'
######################################################################################
# Ajusta configuración YAML
######################################################################################
YOLO_ajusta_yaml(ruta_archivo_yaml, ruta_entrenamiento, ruta_validacion, numero_clases, nombres_clases)
######################################################################################
# Entrena
######################################################################################
YOLO_entrenar_modelo()
######################################################################################
# Testear el modelo
######################################################################################
YOLO_test_de_imagen()


