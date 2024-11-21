from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo
model = YOLO("models/best.pt")  # Asegúrate de usar la ruta correcta a tu modelo exportado

# Función para procesar las detecciones
def detectar_personas_sin_casco(result):
    personas_sin_casco = []
    cascos = []

    # Recorrer todas las detecciones
    for detection in result.boxes.data:  # Detecciones del modelo
        cls, conf, x1, y1, x2, y2 = detection[5], detection[4], *detection[:4]
        label = result.names[int(cls)]
        
        # Convertir tensores a números
        x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())

        # Imprimir las detecciones para depuración
        print(f"Detectado: {label} con confianza: {conf}, coordenadas: {x1, y1, x2, y2}")

        # Clasificar detecciones por clase
        if label == "Person":
            personas_sin_casco.append({
                "bbox": [x1, y1, x2, y2],
                "casco": False  # Inicialmente asumimos que no tiene casco
            })
        elif label == "Helmet":
            cascos.append([x1, y1, x2, y2])  # Guardar coordenadas de los cascos

    # Asociar cascos con personas
    for persona in personas_sin_casco:
        px1, py1, px2, py2 = persona["bbox"]
        for casco in cascos:
            cx1, cy1, cx2, cy2 = casco
            # Comprobar si el casco se encuentra dentro de la persona (superposición simple)
            if (cx1 < px2 and cx2 > px1 and cy1 < py2 and cy2 > py1):
                persona["casco"] = True
                break

    # Filtrar personas sin casco
    personas_sin_casco = [p for p in personas_sin_casco if not p["casco"]]
    return personas_sin_casco

# Función para realizar inferencias en imágenes
def inferir_imagen(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error al cargar la imagen.")
        return

    results = model.predict(image)  # Realizar predicción

    # Procesar resultados
    personas_sin_casco = detectar_personas_sin_casco(results[0])

    # Dibujar detecciones
    for persona in personas_sin_casco:
        x1, y1, x2, y2 = persona["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rojo para sin casco
        cv2.putText(image, "Sin Casco", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(output_path, image)
    print(f"Resultado guardado en {output_path}")

# Ejecución principal
if __name__ == "__main__":
    input_image = "data/input/example.jpg"  # Ruta de la imagen de entrada
    output_image = "data/output/result.jpg"  # Ruta para guardar el resultado
    inferir_imagen(input_image, output_image)
