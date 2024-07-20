from PIL import Image
import os
import random
from shutil import copyfile
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random

# Lista de carpetas que contienen las anotaciones
carpetas = ['Spurious_copper', 'Mouse_bite', 'Open_circuit', 'Missing_hole', 'Spur', 'Short']
# Diccionario que mapea nombres de clases a IDs
class_mapping = {'spurious_copper': 0, 'mouse_bite': 1, 'open_circuit': 2, 'missing_hole': 3, 'spur': 4, 'short': 5}


def resize_images(original_folder, resized_folder,carpetas):
    """
    Redimensiona las imágenes 640x640 píxeles que es el tamaño que consume el modelo en las subcarpetas de la carpeta original y guarda las imágenes redimensionadas en una nueva carpeta.

    Parámetros:
    original_folder (str): La ruta a la carpeta que contiene las subcarpetas con las imágenes originales.
    resized_folder (str): La ruta a la carpeta donde se guardarán las imágenes redimensionadas.

    """
    # Crear la carpeta de destino si no existe
    os.makedirs(resized_folder, exist_ok=True)

    for carpeta in carpetas:
        # Crea las carpetas
        folder_path = os.path.join(original_folder, carpeta)

        # Listar todas las imágenes en la subcarpeta
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Crear la carpeta de destino correspondiente si no existe
        os.makedirs(os.path.join(resized_folder, carpeta), exist_ok=True)

        for image in images:
            # Ruta original
            image_path = os.path.join(folder_path, image)
            img = Image.open(image_path)
            resized_img = img.resize((640, 640))
            # destino
            output_path = os.path.join(resized_folder, carpeta, image)
            # Guardar la imagen redimensionada
            resized_img.save(output_path)
    print("Completado el cambio de tamaño de la imagen y guardado en una nueva carpeta.")

def resize_xml(xml_path, output_path, target_size):
    """
    Redimensiona las anotaciones en un archivo XML para que se ajusten a un tamaño objetivo.

    Parámetros:
    xml_path (str): La ruta del archivo XML original.
    output_path (str): La ruta donde se guardará el archivo XML redimensionado.
    target_size (int): El tamaño objetivo al que se redimensionarán las anotaciones.

    """
    # Cargar y analizar el archivo XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Redimensiona las etiquetas de tamaño en base al nuevo target size
    for size in root.iter('size'):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        size.find('width').text = str(target_size)
        size.find('height').text = str(target_size)

    # Redimensionar las coordenadas de los cuadros delimitadores
    for obj in root.iter('object'):
        for box in obj.iter('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

            xmin = int(xmin * target_size / width)
            ymin = int(ymin * target_size / height)
            xmax = int(xmax * target_size / width)
            ymax = int(ymax * target_size / height)

            box.find('xmin').text = str(xmin)
            box.find('ymin').text = str(ymin)
            box.find('xmax').text = str(xmax)
            box.find('ymax').text = str(ymax)

    # Guardar el archivo XML redimensionado
    tree.write(output_path)

def resize_etiquetas(carpeta_etiquetas, carpeta_etiquetas_resize, target_size, carpetas):
    """
    Redimensiona los archivos XML de anotaciones en las subcarpetas de la carpeta original y guarda los archivos redimensionados en una nueva carpeta.

    Parámetros:
    carpeta_etiquetas (str): La ruta a la carpeta que contiene las etiquetas originales.
    carpeta_etiquetas_resize (str): La ruta a la carpeta donde se guardarán las etiquetas redimensionadas.
    target_size (int): El tamaño objetivo al que se redimensionarán las etiquetas.

    """
    # Crear la carpeta de destino si no existe
    os.makedirs(carpeta_etiquetas_resize, exist_ok=True)

    for carpeta in carpetas:
        # ruta completa de la subcarpeta
        folder_path = os.path.join(carpeta_etiquetas, carpeta)

        # Listar todos los archivos XML en la subcarpeta
        xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
        
        for xml_file in tqdm(xml_files, desc=f"Procesando {carpeta}"):
            # XML original
            xml_path = os.path.join(folder_path, xml_file)

            # Obtener el nombre base del archivo sin la extensión
            base_filename = os.path.splitext(xml_file)[0]
            
            # Ruta completa del archivo XML redimensionado en la carpeta de destino
            output_xml_path = os.path.join(carpeta_etiquetas_resize, f"{base_filename}.xml")
            
            # llama al redimensionar XML
            resize_xml(xml_path, output_xml_path, target_size)

    print("Completado el redimensionamiento de archivos XML y guardado en una nueva carpeta.")

def split_dataset(source_folder, output_folder, train_ratio=0.8, val_ratio=0.2):
    """
    Divide un conjunto de datos en subconjuntos de entrenamiento y validación y copia los archivos a las carpetas correspondientes.

    Parámetros:
    source_folder (str): La ruta a la carpeta que contiene los archivos fuente (imágenes y XML).
    output_folder (str): La ruta a la carpeta donde se guardarán los subconjuntos de entrenamiento y validación.
    train_ratio (float): La proporción de archivos que se asignarán al conjunto de entrenamiento.
    val_ratio (float): La proporción de archivos que se asignarán al conjunto de validación.

    """
    # Crear la carpeta de destino si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Crear subcarpetas de entrenamiento y validación
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_folder, subset), exist_ok=True)

    # Itera sobre los archivos en la carpeta fuente
    for xml_file in os.listdir(source_folder):
        if xml_file.endswith('.xml'):
            # Obtener el nombre base del archivo sin la extensión
            base_filename = os.path.splitext(xml_file)[0]

            # Generar un número aleatorio para decidir el subconjunto (train o val)
            rand_num = random.random()
            if rand_num < train_ratio:
                subset_folder = 'train'
            else:
                subset_folder = 'val'

            # Rutas de los archivos XML y de imagenes
            src_xml = os.path.join(source_folder, xml_file)
            dest_xml = os.path.join(output_folder, subset_folder, f'{base_filename}.xml')
            src_jpg = os.path.join(source_folder, f'{base_filename}.jpg')
            dest_jpg = os.path.join(output_folder, subset_folder, f'{base_filename}.jpg')

            # Copia segun tipo de destino
            copyfile(src_xml, dest_xml)
            copyfile(src_jpg, dest_jpg)

def convert_xml_to_yolo(xml_path, image_width, image_height, class_mapping):
    """
    Convierte anotaciones XML a formato YOLO.

    Parámetros:
    xml_path (str): La ruta del archivo XML.
    image_width (int): El ancho de la imagen.
    image_height (int): La altura de la imagen.
    class_mapping (dict): Un diccionario que mapea nombres de clases a IDs.

    Retorna:
    list: Lista de anotaciones en formato YOLO.
    """
    # Cargar y analizar el archivo XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue

        class_id = class_mapping[class_name]
        bbox = obj.find('bndbox')

        x_center = (float(bbox.find('xmin').text) + float(bbox.find('xmax').text)) / 2.0 / image_width
        y_center = (float(bbox.find('ymin').text) + float(bbox.find('ymax').text)) / 2.0 / image_height
        width = (float(bbox.find('xmax').text) - float(bbox.find('xmin').text)) / image_width
        height = (float(bbox.find('ymax').text) - float(bbox.find('ymin').text)) / image_height

        labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return labels

def create_yolo_labels(source_folder, output_folder, class_mapping):
    """
    Crea etiquetas en formato YOLO a partir de archivos XML redimensionados.

    Parámetros:
    source_folder (str): Carpeta que contiene los archivos XML y las imágenes.
    output_folder (str): Carpeta donde se guardarán las etiquetas YOLO.
    class_mapping (dict): Un diccionario que mapea nombres de clases a IDs.
    """
    # Crear la carpeta de destino si no existe
    os.makedirs(output_folder, exist_ok=True)

    for xml_file in os.listdir(source_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(source_folder, xml_file)

            image_file = os.path.splitext(xml_file)[0] + '.jpg'
            image_path = os.path.join(source_folder.replace('Annotations', 'JPEGImages'), image_file)
            img = Image.open(image_path)
            image_width, image_height = img.size

            labels = convert_xml_to_yolo(xml_path, image_width, image_height, class_mapping)

            output_path = os.path.join(output_folder, os.path.splitext(xml_file)[0] + '.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(labels))

def visualizar_imagen_aleatoria_con_etiquetas(carpeta_imagenes, carpeta_etiquetas):
    """
    Visualiza una imagen aleatoria con sus etiquetas de un conjunto de datos para testear que las imágenes y sus etiquetas estén ajustadas correctamente.

    Parámetros:
    carpeta_imagenes (str): Ruta a la carpeta que contiene las imágenes.
    carpeta_etiquetas (str): Ruta a la carpeta que contiene las etiquetas YOLO en formato .txt.
    """
    # Lista de archivos de imagen en la carpeta de imágenes
    archivos_imagen = [f for f in os.listdir(carpeta_imagenes) if f.endswith('.jpg')]

    # Selección aleatoria de un archivo de imagen
    archivo_imagen_aleatorio = random.choice(archivos_imagen)
    print("Imagen seleccionada aleatoriamente:", archivo_imagen_aleatorio)

    # Ruta completa del archivo de imagen
    ruta_imagen = os.path.join(carpeta_imagenes, archivo_imagen_aleatorio)

    # Carga la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Ruta del archivo de etiquetas correspondiente
    archivo_etiqueta = os.path.splitext(archivo_imagen_aleatorio)[0] + '.txt'
    ruta_etiqueta = os.path.join(carpeta_etiquetas, archivo_etiqueta)

    # Si existe el archivo de etiquetas, se procesan las etiquetas
    if os.path.exists(ruta_etiqueta):
        with open(ruta_etiqueta, 'r') as f:
            lineas = f.readlines()

        # Dibujar las etiquetas en la imagen
        for linea in lineas:
            partes = linea.strip().split()
            id_clase = int(parts[0])
            x_centro, y_centro, ancho, alto = map(float, partes[1:])

            altura_img, ancho_img, _ = imagen.shape
            x, y, w, h = map(int, [x_centro * ancho_img, y_centro * altura_img, ancho * ancho_img, alto * altura_img])
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar la imagen con las etiquetas
    plt.imshow(imagen)
    plt.axis('off')
    plt.show()

######################################################################################
#redimensiona las imágenes
######################################################################################
original_folder = '\\PCB_DATASET\\images'
resized_folder = '\\PCB_resized'
resize_images(original_folder, resized_folder)

######################################################################################
# Ajusta las etiquetas a las imagenes redimesionadas
######################################################################################
carpeta_etiquetas = '\\PCB_DATASET\\Annotations'
carpeta_etiquetas_resize = '\\PCB_resized'
target_size = 640
resize_etiquetas(carpeta_etiquetas, carpeta_etiquetas_resize, target_size)

######################################################################################
# Divide el conjunto en entrenamiento y validación 80/20
######################################################################################
source_folder = '\\PCB_resized'
output_folder = '\\PCB_split'
split_dataset(source_folder, output_folder)

######################################################################################
# Crear etiquetas YOLO para el conjunto de entrenamiento en base a los xml transformados
######################################################################################
create_yolo_labels(
    '\\PCB_split\\train', 
    '\\PCB_split\\train', 
    class_mapping
)

# Crear etiquetas YOLO para el conjunto de validación en base a los xml transformados
create_yolo_labels(
    '\\PCB_split\\val', 
    '\\PCB_split\\val', 
    class_mapping
)

######################################################################################
#Opcional, testeamos con una imagen si las etiquetas quedaron bien
######################################################################################
# Rutas a las carpetas de entrenamiento y etiquetas
carpeta_entrenamiento = '/PCB_split/train'
carpeta_etiquetas = '/PCB_split/train'
# Llamar a la función para visualizar una imagen aleatoria con sus etiquetas
visualizar_imagen_aleatoria_con_etiquetas(carpeta_entrenamiento, carpeta_etiquetas)
