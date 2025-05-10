import cv2
from google.cloud import vision
import io

# ======= 1. Captura de Imagen desde la Webcam =======
def capturar_imagen(nombre_archivo="captura.jpg"):
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        cv2.imwrite(nombre_archivo, frame)
        print("[INFO] Imagen capturada correctamente.")
    else:
        print("[ERROR] No se pudo capturar la imagen.")
    cam.release()
    cv2.destroyAllWindows()
    return nombre_archivo if ret else None

# ======= 2. Detección con Google Vision =======
def detectar_persona_google_vision(archivo_imagen):
    client = vision.ImageAnnotatorClient()

    with io.open(archivo_imagen, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    etiquetas = [obj.name.lower() for obj in objects]
    print("[INFO] Objetos detectados:", etiquetas)

    if "person" in etiquetas:
        print("[RESULTADO] ✅ Persona detectada en la imagen.")
    else:
        print("[RESULTADO] ❌ No se detectó ninguna persona.")

# ======= 3. Main =======
def main():
    nombre_imagen = capturar_imagen()
    if nombre_imagen:
        detectar_persona_google_vision(nombre_imagen)

if __name__ == "__main__":
    main()
