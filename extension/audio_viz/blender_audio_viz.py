"""
blender_audio_viz.py - Visualizador de audio para Blender 5.x.

ANALIZA EL AUDIO EL SOLO. Blender trae su propia libreria de sonido (aud) con
ffmpeg dentro, y numpy: con eso se puede abrir un wav, mp3, flac, ogg o aiff,
analizarlo y montar el visualizador sin salir de aqui ni instalar nada.

Tambien puede cargar un .json ya analizado (por este mismo plugin o por
analiza_audio.py). Sirve para no repetir el analisis de un tema largo, para
procesar muchos archivos de golpe con la herramienta de fuera, o para pasarle a
otra persona el analisis sin el audio.

Lo que hace con los datos:

  1. Un Empty llamado "AudioBands" con una propiedad personalizada por banda
     (band_0 ... band_7), animada fotograma a fotograma. Es el "cerebro":
     un unico sitio donde viven los datos.

  2. Opcionalmente, 8 barras que suben y bajan enganchadas a ese Empty
     mediante drivers.

Como usarlo:
  1. En Blender: pestana Scripting > Abrir > este archivo > Run Script (Alt+P).
  2. En el visor 3D pulsa N y ve a la pestana "Audio Viz".

ATRIBUTOS DISPONIBLES PARA LOS SHADERS
--------------------------------------
El plexus y su objeto de caras llevan dos atributos de dominio PUNTO. Se leen
con un nodo "Attribute" en modo Geometry, usando la salida "Factor":

  av_nivel       0..1   Donde cae el punto en el espectro (0 graves, 1 agudos).
                        Fijo: depende de la geometria, no de la musica.
  av_intensidad  0..1   Cuanto suena la banda de ese punto en este fotograma.
                        Este es el que reacciona al audio.

Al ser de dominio punto, Blender los interpola por la superficie: cada cara sale
degradada entre los valores de sus tres vertices y cada linea entre los de sus
dos puntas. Por eso una cara puede "tomar el color de la intensidad de sus
vertices" sin hacer nada raro: basta con enchufar av_intensidad a una rampa.

Probado en Blender 5.0.1.
"""

import colorsys
import json
import math
import os
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import Operator, Panel, PropertyGroup

try:
    import numpy as np
except ImportError:  # Blender siempre lo trae, pero por si acaso
    np = None

NOMBRE_EMPTY = "AudioBands"      # nombre de las fuentes de versiones anteriores
PREFIJO_FUENTE = "AV_Audio"      # las nuevas se llaman AV_Audio_<nombre del json>
NOMBRE_COLECCION = "AudioViz"
PREFIJO_BANDA = "band_"
PREFIJO_MATERIAL = "AV_Banda_"

# Propiedades internas donde el Empty guarda los valores ORIGINALES del JSON,
# sin suavizar. No las toques a mano: son la copia maestra desde la que se
# recalcula la animacion cada vez que mueves los deslizadores de suavizado.
CLAVE_CRUDO = "av_raw"
CLAVE_FRAMES = "av_frames"
CLAVE_BANDAS = "av_bandas"

# Estereo. El canal MONO es el de siempre y no se toca, para que nada de lo ya
# hecho cambie; los otros dos solo existen si el archivo traia dos canales.
CLAVE_CRUDO_IZQ = "av_raw_izq"
CLAVE_CRUDO_DER = "av_raw_der"
PREFIJO_BANDA_IZQ = "band_izq_"
PREFIJO_BANDA_DER = "band_der_"

CANALES = (
    ('MONO', "Mono", "Los dos canales mezclados, como hasta ahora"),
    ('IZQ', "Solo el izquierdo", "Reacciona nada mas al canal izquierdo"),
    ('DER', "Solo el derecho", "Reacciona nada mas al canal derecho"),
    ('ESTEREO', "Estereo", "El lado izquierdo del objeto sigue al canal izquierdo "
                           "y el derecho al derecho"),
)

# Preset 1: ecualizador de LEDs
PREFIJO_LED = "AV_Led"
NOMBRE_MAT_LED = "AV_Material_Led"

# Preset 2: plexus
NOMBRE_PLEXUS = "AV_Plexus"
NOMBRE_GN_PLEXUS = "AV_Plexus_Nodos"
NOMBRE_MAT_PLEXUS = "AV_Material_Plexus"
SUFIJO_CARAS = "_Caras"
NOMBRE_MAT_CARAS = "AV_Material_Caras"

# Preset: el compas visto en cubos
PREFIJO_PULSO_VIS = "AV_Pulso"
NOMBRE_MAT_PULSO = "AV_Material_Pulso"

# Preset 3: paisaje que avanza
NOMBRE_PAISAJE = "AV_Paisaje"
NOMBRE_MAT_PAISAJE = "AV_Material_Paisaje"
CLAVE_FIRMA_PAISAJE = "av_paisaje_firma"

# Preset 4: enjambre orbital
NOMBRE_ENJAMBRE = "AV_Enjambre"
NOMBRE_GN_ENJAMBRE = "AV_Enjambre_Nodos"
NOMBRE_MAT_ENJAMBRE = "AV_Material_Enjambre"
CLAVE_FIRMA_ENJAMBRE = "av_enjambre_firma"
CLAVE_ENJ_RAD = "av_enj_rad"
CLAVE_ENJ_RADN = "av_enj_radn"
CLAVE_ENJ_ANG = "av_enj_ang"
CLAVE_ENJ_ALT = "av_enj_alt"
CLAVE_ENJ_BANDA = "av_enj_banda"


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def leer_json(ruta):
    """Devuelve (lista_de_fotogramas, matriz [fotograma][banda])."""
    with open(bpy.path.abspath(ruta), "r", encoding="utf-8") as f:
        datos = json.load(f)

    if not datos:
        raise ValueError("El JSON esta vacio.")

    # Las claves son cadenas ("0", "1", "2"...) porque JSON no admite claves
    # numericas. Las ordenamos como numeros, no como texto: si no, el
    # fotograma 10 iria antes que el 2.
    claves = sorted(datos.keys(), key=int)
    n_bandas = len(datos[claves[0]])

    for k in claves:
        if len(datos[k]) != n_bandas:
            raise ValueError(f"El fotograma {k} tiene {len(datos[k])} bandas y se esperaban {n_bandas}.")

    fotogramas = [int(k) for k in claves]
    matriz = [datos[k] for k in claves]
    return fotogramas, matriz, n_bandas


# ---------------------------------------------------------------------------
# ANALISIS DE AUDIO DENTRO DE BLENDER
# ---------------------------------------------------------------------------
# Blender trae su propia libreria de audio ('aud') y ffmpeg, asi que puede abrir
# wav, mp3, flac, ogg o aiff y darnos las muestras. Con eso y numpy -que tambien
# viene incluido- se puede hacer aqui el mismo analisis que hace el programa de
# fuera, y no depender de nada externo.
#
# El calculo es identico al de analiza_audio.py; lo unico distinto es que aqui
# las FFT se hacen por bloques en vez de una a una, que es varias veces mas
# rapido. Los .json que genera el programa de fuera se siguen pudiendo cargar:
# es util para no repetir el analisis de un tema largo, o para pasarselo a otro.

EXTENSIONES_AUDIO = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".aiff",
                     ".aif", ".aac", ".m4a", ".wma", ".w64", ".ac3", ".mp4", ".mkv")


def leer_audio(ruta):
    """Devuelve (mono, izquierda, derecha, frecuencia).

    Los canales sueltos son None si el archivo es mono. El mono NO se deduce
    luego de los otros dos: se mezcla antes de analizar, que es como se hacia
    siempre, para que los proyectos de antes sigan dando exactamente lo mismo.
    """
    import aud
    sonido = aud.Sound.file(str(ruta))
    datos = sonido.data()
    if datos is None or len(datos) == 0:
        raise ValueError("el archivo no contiene audio")
    frec = int(round(float(sonido.specs[0])))

    if datos.ndim > 1 and datos.shape[1] >= 2:
        izq = np.ascontiguousarray(datos[:, 0], dtype=np.float64)
        der = np.ascontiguousarray(datos[:, 1], dtype=np.float64)
        mono = np.ascontiguousarray(datos.mean(axis=1), dtype=np.float64)
        return mono, izq, der, frec

    plano = datos[:, 0] if datos.ndim > 1 else datos
    return np.ascontiguousarray(plano, dtype=np.float64), None, None, frec


# ---------------------------------------------------------------------------
# EL AUDIO EN EL SECUENCIADOR
# ---------------------------------------------------------------------------
# Meter el archivo en el Video Sequencer es lo que permite OIRLO mientras mueves
# el cursor por la linea de tiempo, que es como se comprueba de verdad si la
# animacion va a tiempo.
#
# OJO con la API: en Blender 5.0 la coleccion se llama `strips`; el viejo
# `sequences` ya no existe. Y una coleccion vacia es "falsa" en Python, asi que
# no se puede elegir entre las dos con un `or`.

PREFIJO_TIRA = "AV_"


def tiras_de(escena, crear=False):
    se = escena.sequence_editor
    if se is None:
        if not crear:
            return None
        se = escena.sequence_editor_create()
    return se.strips if hasattr(se, "strips") else se.sequences


def canal_libre(tiras):
    ocupados = {t.channel for t in tiras}
    canal = 1
    while canal in ocupados:
        canal += 1
    return canal


def tira_de_fuente(escena, fuente):
    """La tira de audio de esta fuente, si sigue existiendo."""
    tiras = tiras_de(escena)
    nombre = fuente.audioviz_audio.tira_sonido
    if tiras is None or not nombre:
        return None
    return tiras.get(nombre)


def quitar_tira_sonido(escena, fuente):
    """Quita la tira de esta fuente y su sonido si no lo usa nadie mas."""
    aud = fuente.audioviz_audio
    tiras = tiras_de(escena)
    if tiras is None or not aud.tira_sonido:
        aud.tira_sonido = ""
        return False

    tira = tiras.get(aud.tira_sonido)
    aud.tira_sonido = ""
    if tira is None:
        return False

    sonido = getattr(tira, "sound", None)
    tiras.remove(tira)
    if sonido is not None and sonido.users == 0:
        bpy.data.sounds.remove(sonido)
    return True


def poner_tira_sonido(escena, fuente, ruta, fotograma_inicio):
    """Mete el audio en el secuenciador y deja el scrub activado."""
    quitar_tira_sonido(escena, fuente)
    tiras = tiras_de(escena, crear=True)

    aud = fuente.audioviz_audio
    nombre = f"{PREFIJO_TIRA}{etiqueta_fuente(fuente)}"
    tira = tiras.new_sound(name=nombre, filepath=str(ruta),
                           channel=canal_libre(tiras),
                           frame_start=int(fotograma_inicio))
    aud.tira_sonido = tira.name
    tira.mute = not aud.oir_audio

    # Sin esto solo se oye al reproducir, no al arrastrar el cursor, que es
    # justo cuando hace falta para cuadrar la animacion.
    escena.use_audio_scrub = True
    return tira


def _al_cambiar_oir_audio(self, contexto):
    """La casilla silencia la pista; no la crea ni la borra.

    Antes esta opcion solo decidia si al analizar se metia el audio o no, y era
    un mal diseno: desactivarla no callaba lo que ya sonaba, y activarla
    obligaba a volver a cargar el archivo. Ahora el audio se mete siempre y esto
    es un mute de verdad. Como unica cortesia, si la pista no esta (porque la
    quitaste a mano) y vuelves a activar la casilla, se recupera sola.
    """
    fuente = self.id_data
    if not es_fuente(fuente):
        return
    escena = contexto.scene

    tira = tira_de_fuente(escena, fuente)
    if tira is None and self.oir_audio:
        ruta = bpy.path.abspath(self.ruta_audio)
        if ruta and os.path.isfile(ruta):
            marcos = fuente.get(CLAVE_FRAMES)
            inicio = int(marcos[0]) if marcos else escena.frame_start
            try:
                tira = poner_tira_sonido(escena, fuente, ruta, inicio)
            except Exception as e:
                print(f"Audio Viz: no he podido recuperar la pista de audio: {e}")
    if tira is None:
        return

    tira.mute = not self.oir_audio
    if self.oir_audio:
        escena.use_audio_scrub = True


def bordes_de_bandas(n_bandas, f_min, f_max):
    """Reparto logaritmico: cada banda cubre las mismas octavas que la anterior.

    El oido percibe asi la frecuencia. En lineal, 7 de 8 bandas caerian en agudos
    donde apenas hay contenido y los graves quedarian apelmazados en una sola.
    """
    return np.logspace(math.log10(f_min), math.log10(f_max), n_bandas + 1)


def analizar_muestras(x, frec, fps, n_bandas, tam_ventana, f_min, f_max, progreso=None):
    """FFT por ventanas sincronizada a los fotogramas. Devuelve (dB, bordes)."""
    # La FFT supone que el trozo se repite en bucle: si se corta a hachazo, el
    # salto genera frecuencias fantasma. La campana de Hann atenua los extremos.
    ventana = np.hanning(tam_ventana)
    frecuencias = np.fft.rfftfreq(tam_ventana, d=1.0 / frec)

    bordes = bordes_de_bandas(n_bandas, f_min, f_max)
    por_banda = []
    for i in range(n_bandas):
        sel = np.where((frecuencias >= bordes[i]) & (frecuencias < bordes[i + 1]))[0]
        if sel.size == 0:
            # Banda mas estrecha que la resolucion de la FFT (pasa en los graves):
            # cogemos el bin mas cercano a su centro.
            centro = math.sqrt(bordes[i] * bordes[i + 1])
            sel = np.array([int(np.argmin(np.abs(frecuencias - centro)))])
        por_banda.append(sel)

    n_fotogramas = int(math.floor(len(x) / frec * fps)) + 1
    mitad = tam_ventana // 2
    db = np.zeros((n_fotogramas, n_bandas), dtype=np.float64)

    # Por bloques: hacer las 4000 ventanas de un tema de 3 minutos de una vez
    # pediria ~70 MB de golpe sin necesidad.
    desplazamiento = np.arange(tam_ventana)
    bloque = 256
    for a in range(0, n_fotogramas, bloque):
        b = min(a + bloque, n_fotogramas)
        # La ventana va CENTRADA en el instante del fotograma: si empezara ahi,
        # cada golpe caeria medio fotograma tarde.
        centros = np.round(np.arange(a, b) / fps * frec).astype(np.int64)
        indices = (centros - mitad)[:, None] + desplazamiento[None, :]
        dentro = (indices >= 0) & (indices < len(x))
        trozos = np.where(dentro, x[np.clip(indices, 0, len(x) - 1)], 0.0)

        espectros = np.abs(np.fft.rfft(trozos * ventana, axis=1))
        for i, sel in enumerate(por_banda):
            # RMS de los bins: la energia media, no la suma, para que una banda
            # ancha no gane solo por contener mas casillas.
            energia = np.sqrt((espectros[:, sel] ** 2).mean(axis=1))
            db[a:b, i] = 20.0 * np.log10(energia + 1e-12)

        if progreso is not None:
            progreso(b, n_fotogramas)

    return db, bordes


def normalizar_estereo(db_izq, db_der, rango_db, modo):
    """Normaliza los dos canales CON EL MISMO TECHO.

    Si cada uno se normalizara por su cuenta, un canal flojo se estiraria hasta
    tocar el techo y la imagen estereo se perderia: todo sonaria centrado. El
    techo compartido es lo que conserva que un lado suene mas que el otro.
    """
    pico = max(float(db_izq.max()), float(db_der.max()))
    if modo == 'GLOBAL':
        techo = np.full(db_izq.shape[1], pico)
    else:
        techo = np.maximum(db_izq.max(axis=0), db_der.max(axis=0))

    suelo = techo - rango_db
    ancho = np.maximum(techo - suelo, 1e-9)
    salida = []
    for db in (db_izq, db_der):
        y = (db - suelo) / ancho
        if modo == 'BANDA':
            y[:, techo < (pico - 80.0)] = 0.0
        salida.append(np.clip(y, 0.0, 1.0))
    return salida[0], salida[1]


def normalizar_db(db, rango_db, modo):
    """De decibelios a 0..1, como los niveles de un programa de color."""
    pico_global = float(db.max())
    if modo == 'GLOBAL':
        # Un unico techo: se conserva la relacion real entre graves y agudos.
        techo = np.full(db.shape[1], pico_global)
    else:
        # Cada banda contra su propio pico: los agudos, mucho mas debiles, se ven
        # igual de vivos que el bombo. Es lo que suele querer un visualizador.
        techo = db.max(axis=0)

    suelo = techo - rango_db
    y = (db - suelo) / np.maximum(techo - suelo, 1e-9)

    if modo == 'BANDA':
        # Una banda muda todo el tema (nada por encima de 12 kHz, por ejemplo)
        # convertiria su ruido de fondo en un baile. La dejamos a cero.
        y[:, techo < (pico_global - 80.0)] = 0.0

    return np.clip(y, 0.0, 1.0)


# ---------------------------------------------------------------------------
# VER EL ANALISIS
# ---------------------------------------------------------------------------
# Un espectrograma del tema entero como imagen: el tiempo de izquierda a
# derecha, las bandas de grave (abajo) a agudo (arriba) y el color segun cuanto
# suena. Se dibuja leyendo las CURVAS, no los valores en crudo, asi que refleja
# el ataque y la caida que tengas puestos: es la unica forma de ver que hace ese
# ajuste sin ir probando a ciegas.
#
# Encima van los pulsos del compas, si esta detectado. Si las marcas caen sobre
# las columnas brillantes, el tempo esta bien; si se van separando, no.

NOMBRE_IMAGEN = "AV_Analisis"
ANCHO_MAX_ESPECTRO = 2048

# Rampa de espectrograma: negro -> azul -> cian -> amarillo -> blanco. La misma
# que la herramienta de fuera, y la que usa cualquier analizador.
_PARADAS_POS = np.array([0.0, 0.25, 0.50, 0.75, 1.0]) if np is not None else None
_PARADAS_COL = (np.array([[0.02, 0.03, 0.08],
                          [0.08, 0.20, 0.55],
                          [0.00, 0.67, 0.75],
                          [0.96, 0.82, 0.24],
                          [1.00, 1.00, 0.96]]) if np is not None else None)


def colores_espectro(v):
    """De valores 0..1 a colores RGB, todo de una vez."""
    v = np.clip(v, 0.0, 1.0)
    return np.stack([np.interp(v, _PARADAS_POS, _PARADAS_COL[:, c]) for c in range(3)],
                    axis=-1)


# Guardamos el dibujo base para poder repintar solo la linea del fotograma sin
# recalcularlo todo: mover el cursor cuesta asi menos de un milisegundo.
_espectros = {}


def generar_espectro(fuente, canal='MONO', alto_banda=18):
    """Crea o actualiza la imagen del analisis. Devuelve (imagen, aviso)."""
    if np is None:
        return None, "hace falta numpy"
    crudos = leer_crudos(fuente, canal_util(fuente, canal))
    if crudos is None:
        return None, "esta fuente no guarda los valores originales"
    fotogramas, _matriz, n_bandas = crudos
    curvas = curvas_de_bandas(fuente, n_bandas, canal_util(fuente, canal))
    if curvas is None:
        return None, "no encuentro las curvas de las bandas"

    primero, ultimo = int(fotogramas[0]), int(fotogramas[-1])
    total = ultimo - primero + 1
    ancho = min(total, ANCHO_MAX_ESPECTRO)
    # Si el tema es largo, cada columna resume varios fotogramas quedandose con
    # el pico: asi un golpe corto no desaparece al reducir.
    bordes = np.linspace(0, total, ancho + 1).astype(int)

    valores = np.empty((n_bandas, total), dtype=np.float64)
    for b, fc in enumerate(curvas):
        for i in range(total):
            valores[b, i] = fc.evaluate(primero + i)

    reducido = np.empty((n_bandas, ancho), dtype=np.float64)
    for i in range(ancho):
        a, z = bordes[i], max(bordes[i + 1], bordes[i] + 1)
        reducido[:, i] = valores[:, a:z].max(axis=1)

    # La banda 0 abajo: en la imagen la fila 0 es la de abajo, asi que sale sola.
    alto = n_bandas * alto_banda
    rgb = colores_espectro(reducido)                       # (bandas, ancho, 3)
    lienzo = np.repeat(rgb, alto_banda, axis=0)            # cada banda, mas gruesa

    # Una raya fina entre bandas, para poder contarlas.
    lienzo[::alto_banda, :, :] *= 0.45

    # Los pulsos del compas, en la franja de abajo.
    if tiene_compas(fuente):
        aud = fuente.audioviz_audio
        periodo = aud.fps * 60.0 / max(aud.bpm, 1e-6)
        ppc = max(int(aud.pulsos_por_compas), 1)
        k = 0
        while True:
            f = aud.desfase_compas + k * periodo
            if f > total:
                break
            col = int(f / max(total, 1) * ancho)
            if 0 <= col < ancho:
                fuerte = (k % ppc) == 0
                alto_marca = 7 if fuerte else 4
                tono = 1.0 if fuerte else 0.55
                lienzo[:alto_marca, col, :] = tono
            k += 1
            if k > 20000:
                break

    nombre = f"{NOMBRE_IMAGEN}_{etiqueta_fuente(fuente)}"
    img = bpy.data.images.get(nombre)
    if img is None or img.size[0] != ancho or img.size[1] != alto:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(nombre, width=ancho, height=alto, alpha=False)

    # Esta imagen no la usa ningun material ni ningun nodo, asi que para Blender
    # no la usa nadie y la tiraria al guardar. El "usuario falso" es justo para
    # esto: le dice que la conserve aunque este suelta. La quita el boton de la X.
    img.use_fake_user = True

    _espectros[nombre] = (lienzo, primero, ultimo)
    pintar_espectro(img, lienzo, None)
    return img, ""


def pintar_espectro(img, lienzo, columna):
    """Vuelca el lienzo a la imagen, con la marca del fotograma si toca."""
    alto, ancho = lienzo.shape[0], lienzo.shape[1]
    px = np.empty((alto, ancho, 4), dtype=np.float32)
    px[:, :, :3] = lienzo
    px[:, :, 3] = 1.0
    if columna is not None and 0 <= columna < ancho:
        px[:, columna, :3] = np.minimum(px[:, columna, :3] * 0.4 + 0.6, 1.0)
    img.pixels.foreach_set(px.ravel())
    img.update()


def actualizar_marca_espectro(escena):
    """Mueve la linea del fotograma. Barato: no recalcula el espectrograma."""
    for nombre, (lienzo, primero, ultimo) in list(_espectros.items()):
        img = bpy.data.images.get(nombre)
        if img is None or tuple(img.size) != (lienzo.shape[1], lienzo.shape[0]):
            # La imagen ya no esta, o es otra distinta que se llama igual
            # (por ejemplo al abrir otro .blend). El lienzo no le vale.
            _espectros.pop(nombre, None)
            continue
        total = max(ultimo - primero + 1, 1)
        frac = (escena.frame_current - primero) / total
        columna = int(frac * lienzo.shape[1])
        pintar_espectro(img, lienzo, columna if 0 <= columna < lienzo.shape[1] else None)


# ---------------------------------------------------------------------------
# FUENTES DE AUDIO
# ---------------------------------------------------------------------------
# Una "fuente" es un Empty que lleva dentro las bandas de UN archivo de audio.
# Puede haber tantas como quieras en la escena, cada una con su propio .json,
# su propio suavizado y sus propios fps.

def es_fuente(ob):
    if ob is None or ob.type != 'EMPTY':
        return False
    aud = getattr(ob, "audioviz_audio", None)
    if aud is not None and aud.es_fuente:
        return True
    # Escenas hechas con la version anterior, cuando solo habia un Empty
    # llamado "AudioBands": si lleva datos de bandas, cuenta como fuente.
    return CLAVE_BANDAS in ob


def fuentes_de_la_escena(escena):
    return [o for o in escena.objects if es_fuente(o)]


def fuente_activa(escena):
    """La fuente elegida en el panel; si no hay ninguna elegida, la primera."""
    elegida = escena.audioviz.fuente
    if es_fuente(elegida):
        return elegida
    lista = fuentes_de_la_escena(escena)
    return lista[0] if lista else None


def etiqueta_fuente(fuente):
    """Nombre corto de la fuente, para bautizar las barras y los LEDs."""
    nombre = fuente.name
    for prefijo in (PREFIJO_FUENTE + "_", PREFIJO_FUENTE):
        if nombre.startswith(prefijo):
            return nombre[len(prefijo):] or "audio"
    return nombre


def _poll_fuente(self, ob):
    # OJO: esto solo filtra el desplegable de la interfaz. Blender NO impide
    # asignar otra cosa desde codigo, asi que ademas hay que comprobarlo al usar.
    return es_fuente(ob)


def obtener_coleccion(escena):
    """La coleccion 'AudioViz', creandola si hace falta."""
    col = bpy.data.collections.get(NOMBRE_COLECCION)
    if col is None:
        col = bpy.data.collections.new(NOMBRE_COLECCION)
        escena.collection.children.link(col)
    elif col.name not in {c.name for c in escena.collection.children_recursive}:
        escena.collection.children.link(col)
    return col


def obtener_bolsa(ob, crear=True):
    """El 'channelbag' donde viven las F-Curves de este objeto.

    OJO: en Blender 4.4 cambio el sistema de animacion ("slotted actions") y
    en 5.0 el viejo `action.fcurves` YA NO EXISTE. Ahora la jerarquia es:
        Accion > Slot (a que objeto afecta) > Capa > Tira > Channelbag > F-Curves
    Si copias codigo de tutoriales de Blender 3.x te dara AttributeError.
    """
    ad = ob.animation_data
    if ad is None:
        if not crear:
            return None
        ad = ob.animation_data_create()

    if ad.action is None:
        if not crear:
            return None
        ad.action = bpy.data.actions.new(f"{ob.name}_Action")
    accion = ad.action

    if ad.action_slot is None:
        if not crear:
            return None
        ad.action_slot = accion.slots.new(id_type='OBJECT', name=ob.name)

    if not accion.layers:
        if not crear:
            return None
        accion.layers.new("Capa")
    capa = accion.layers[0]

    if not capa.strips:
        if not crear:
            return None
        capa.strips.new(type='KEYFRAME')

    return capa.strips[0].channelbag(ad.action_slot, ensure=crear)


def crear_curvas(ob, rutas_datos, borrar_todo=False):
    """Curvas vacias para esas propiedades, respetando las demas.

    `borrar_todo` solo se usa al reimportar un audio: entonces si interesa
    empezar de cero. En cualquier otro caso las curvas del compas y las de las
    bandas conviven en la misma accion y no deben pisarse.
    """
    if borrar_todo:
        ad = ob.animation_data
        if ad is not None and ad.action is not None:
            vieja = ad.action
            ad.action = None
            if vieja.users == 0:
                bpy.data.actions.remove(vieja)

    bolsa = obtener_bolsa(ob)
    existentes = {fc.data_path: fc for fc in bolsa.fcurves}
    curvas = []
    for ruta in rutas_datos:
        antigua = existentes.get(ruta)
        if antigua is not None:
            bolsa.fcurves.remove(antigua)
        curvas.append(bolsa.fcurves.new(ruta, index=0))
    return curvas


def borrar_curvas(ob, rutas_datos):
    bolsa = obtener_bolsa(ob, crear=False)
    if bolsa is None:
        return
    for fc in list(bolsa.fcurves):
        if fc.data_path in rutas_datos:
            bolsa.fcurves.remove(fc)


def rellenar_curva(fcurve, fotogramas, valores, interpolacion=1):
    """Escribe todas las claves de golpe.

    Insertarlas una a una con keyframe_insert() funciona pero es ~37 veces mas
    lento: en un tema de 3 minutos son 34.000 claves y se nota. `foreach_set`
    vuelca la lista entera de un tiron a la estructura interna de Blender.
    """
    n = len(fotogramas)
    fcurve.keyframe_points.add(count=n)

    # 'co' son las coordenadas de cada clave en la grafica: [x=fotograma, y=valor].
    plano = [0.0] * (n * 2)
    plano[0::2] = [float(f) for f in fotogramas]
    plano[1::2] = [float(v) for v in valores]
    fcurve.keyframe_points.foreach_set("co", plano)

    # Interpolacion LINEAL (el codigo 1) por defecto. Con una clave en cada
    # fotograma, la bezier se pasaria de frenada entre claves y sacaria valores
    # por encima de 1 o por debajo de 0. El codigo 0 es CONSTANTE, a saltos.
    fcurve.keyframe_points.foreach_set("interpolation", [interpolacion] * n)
    fcurve.update()


# ---------------------------------------------------------------------------
# SUAVIZADO EN VIVO (ataque y caida)
# ---------------------------------------------------------------------------
# Esto es un limitador de pendiente, lo mismo que el ataque y el release de un
# compresor o la inercia de la aguja de un vumetro: el valor no puede subir ni
# bajar mas de cierta cantidad por fotograma.
#
# El usuario lo ajusta en SEGUNDOS (cuanto tarda un pico en recorrer todo el
# recorrido de 1.0 a 0.0), que es como se piensa en audio: mas segundos = cola
# mas larga. Internamente hay que convertirlo a "unidades por fotograma", que es
# lo contrario: mas segundos = tasa mas pequena.
#
# No se puede hacer con un driver: cada fotograma necesita saber cuanto valia el
# anterior, y eso en Blender seria una dependencia circular. Por eso lo que
# hacemos es reescribir las claves de animacion, que cuesta ~10 ms y por tanto
# se siente instantaneo aunque tecnicamente sea un "rehorneado".

def tasa_por_fotograma(segundos, fps):
    """De 'segundos en recorrer 0..1' a 'unidades por fotograma'. 0 = instantaneo."""
    if segundos <= 0.0:
        return 0.0
    return 1.0 / (segundos * max(float(fps), 1.0))


def suavizar(valores, ataque, caida):
    """Aplica el limitador de pendiente a una banda. 0 = sin limite (instantaneo)."""
    if ataque <= 0.0 and caida <= 0.0:
        return list(valores)

    subida = math.inf if ataque <= 0.0 else ataque
    bajada = math.inf if caida <= 0.0 else caida

    salida = []
    anterior = valores[0]  # el primer fotograma pasa tal cual: no hay pasado
    for v in valores:
        if v > anterior + subida:
            v = anterior + subida
        elif v < anterior - bajada:
            v = anterior - bajada
        anterior = v
        salida.append(v)
    return salida


CLAVES_CANAL = {'MONO': CLAVE_CRUDO, 'IZQ': CLAVE_CRUDO_IZQ, 'DER': CLAVE_CRUDO_DER}
PREFIJOS_CANAL = {'MONO': PREFIJO_BANDA, 'IZQ': PREFIJO_BANDA_IZQ, 'DER': PREFIJO_BANDA_DER}


def guardar_crudos(empty, fotogramas, matrices, n_bandas):
    """Guarda dentro del .blend los valores originales, canal por canal.

    Asi el suavizado se puede cambiar despues sin volver a tocar el archivo, e
    incluso aunque le pases el .blend a otra persona sin el audio.

    `matrices` es {'MONO': [...], 'IZQ': [...], 'DER': [...]}; los canales que
    falten se borran, que es lo que toca al reemplazar un estereo por un mono.
    """
    empty[CLAVE_BANDAS] = n_bandas
    empty[CLAVE_FRAMES] = [int(f) for f in fotogramas]
    for canal, clave in CLAVES_CANAL.items():
        matriz = matrices.get(canal)
        if matriz is None:
            if clave in empty:
                del empty[clave]
            continue
        # Una sola lista plana [f0b0, f0b1, ..., f1b0, ...] en vez de una por banda.
        empty[clave] = [float(v) for fila in matriz for v in fila]


def leer_crudos(empty, canal='MONO'):
    """(fotogramas, matriz, n_bandas) de ese canal, o None si no esta guardado."""
    clave = CLAVES_CANAL.get(canal, CLAVE_CRUDO)
    if clave not in empty or CLAVE_FRAMES not in empty or CLAVE_BANDAS not in empty:
        return None
    n_bandas = int(empty[CLAVE_BANDAS])
    fotogramas = [int(f) for f in empty[CLAVE_FRAMES]]
    plano = list(empty[clave])
    if n_bandas <= 0 or len(plano) != len(fotogramas) * n_bandas:
        return None
    matriz = [plano[i * n_bandas:(i + 1) * n_bandas] for i in range(len(fotogramas))]
    return fotogramas, matriz, n_bandas


def bandas_de(fuente, canal='MONO'):
    """Indices de banda de ese canal, en orden.

    OJO con el prefijo: 'band_izq_0' TAMBIEN empieza por 'band_', asi que no
    vale con mirar el principio. Solo cuentan las claves cuyo resto es un
    numero.
    """
    prefijo = PREFIJOS_CANAL.get(canal, PREFIJO_BANDA)
    salida = []
    for clave in fuente.keys():
        if clave.startswith(prefijo):
            resto = clave[len(prefijo):]
            if resto.isdigit():
                salida.append(int(resto))
    return sorted(salida)


def es_estereo(fuente):
    """Si esta fuente guarda los dos canales por separado."""
    return (fuente is not None and CLAVE_CRUDO_IZQ in fuente
            and CLAVE_CRUDO_DER in fuente)


def canales_de(fuente):
    """Los canales que de verdad se pueden usar con esta fuente."""
    return ('MONO', 'IZQ', 'DER') if es_estereo(fuente) else ('MONO',)


def curvas_de_bandas(empty, n_bandas, canal='MONO'):
    """Las F-Curves ya existentes de ese canal, o None si no estan todas."""
    bolsa = obtener_bolsa(empty, crear=False)
    if bolsa is None:
        return None
    prefijo = PREFIJOS_CANAL.get(canal, PREFIJO_BANDA)
    mapa = {fc.data_path: fc for fc in bolsa.fcurves}
    curvas = [mapa.get(f'["{prefijo}{i}"]') for i in range(n_bandas)]
    return None if any(c is None for c in curvas) else curvas


def aplicar_suavizado(empty, ataque_seg, caida_seg, fps):
    """Recalcula la animacion desde los valores originales. Devuelve True si pudo.

    `ataque_seg` y `caida_seg` van en SEGUNDOS; aqui se traducen a la tasa por
    fotograma que necesita suavizar(). Se aplica igual a los tres canales: seria
    raro que el izquierdo tuviera una cola distinta que el derecho.
    """
    ataque = tasa_por_fotograma(ataque_seg, fps)
    caida = tasa_por_fotograma(caida_seg, fps)
    hecho = False

    for canal in ('MONO', 'IZQ', 'DER'):
        crudos = leer_crudos(empty, canal)
        if crudos is None:
            continue
        fotogramas, matriz, n_bandas = crudos
        prefijo = PREFIJOS_CANAL[canal]

        for i in range(n_bandas):
            clave = f"{prefijo}{i}"
            if clave not in empty:
                empty[clave] = 0.0
                empty.id_properties_ui(clave).update(
                    min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, default=0.0)

        curvas = curvas_de_bandas(empty, n_bandas, canal)
        reconstruir = curvas is None or len(curvas[0].keyframe_points) != len(fotogramas)
        if reconstruir:
            # Sin borrar_todo: aqui solo se rehacen las curvas de este canal. Las
            # del compas y las de los otros canales viven en la misma accion y no
            # tienen por que perderse.
            rutas = [f'["{prefijo}{i}"]' for i in range(n_bandas)]
            curvas = crear_curvas(empty, rutas)

        for i, fc in enumerate(curvas):
            valores = suavizar([fila[i] for fila in matriz], ataque, caida)
            if reconstruir:
                rellenar_curva(fc, fotogramas, valores)
            else:
                # Camino rapido: los fotogramas no cambian, solo la altura de
                # cada clave. Leemos las coordenadas, pisamos las 'y' y vuelven.
                n = len(valores)
                plano = [0.0] * (n * 2)
                fc.keyframe_points.foreach_get("co", plano)
                plano[1::2] = valores
                fc.keyframe_points.foreach_set("co", plano)
                fc.update()
        hecho = True

    return hecho


# ---------------------------------------------------------------------------
# COMPAS: SACAR EL TEMPO DEL PROPIO JSON
# ---------------------------------------------------------------------------
# No tenemos el audio, solo las 8 bandas a 24 fps. Aun asi se puede sacar el
# tempo, y es lo que hace cualquier detector de ritmo:
#
#   1. Funcion de novedad: donde SUBE la energia de golpe hay un ataque. Se mira
#      cuanto crece cada banda respecto al fotograma anterior y se suma.
#   2. Tempo: se prueban muchos BPM y para cada uno se "dobla" la novedad sobre
#      un solo periodo (como enrollar la cancion en un carrete). Si el BPM es el
#      correcto, todos los golpes caen en el mismo sitio del carrete y aparece un
#      pico limpio; si no, se reparten y queda plano.
#   3. Fase: la posicion de ese pico dentro del periodo dice donde cae el primer
#      golpe.
#
# La resolucion no la limitan los 24 fps: como se mide sobre la cancion entera,
# un error minusculo de BPM desplaza los ultimos golpes y se nota. Por eso se
# puede afinar a decimas de BPM aunque cada fotograma dure 42 ms.

CLAVE_PULSO = "pulso"
CLAVE_FASE_PULSO = "fase_pulso"
CLAVE_FASE_COMPAS = "fase_compas"
CLAVE_NUM_PULSO = "numero_pulso"
CLAVES_COMPAS = (CLAVE_PULSO, CLAVE_FASE_PULSO, CLAVE_FASE_COMPAS, CLAVE_NUM_PULSO)
RUTAS_COMPAS = tuple(f'["{c}"]' for c in CLAVES_COMPAS)
PREFIJO_MARCADOR = "AV_"


def funcion_novedad(matriz, fps, banda_min, banda_max):
    """Cuanta energia GANA el sonido en cada fotograma: ahi hay un golpe."""
    m = np.asarray(matriz, dtype=np.float64)
    if m.ndim != 2 or len(m) < 4:
        return None

    b0 = max(0, min(int(banda_min), m.shape[1] - 1))
    b1 = max(b0, min(int(banda_max), m.shape[1] - 1))
    m = m[:, b0:b1 + 1]

    # Solo las subidas. Una bajada no es un ataque.
    subida = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0.0).sum(axis=1)

    # Le quitamos la tendencia lenta restando su media movil de ~1 segundo, para
    # que un tema que gana intensidad no tape los golpes del principio.
    ancho = max(3, int(round(fps)))
    nucleo = np.ones(ancho) / ancho
    suave = np.convolve(subida, nucleo, mode="same")
    novedad = np.maximum(subida - suave, 0.0)

    pico = float(novedad.max())
    return novedad / pico if pico > 1e-12 else None


def _fase_por_doblado(novedad, indices, periodo, casillas):
    """Donde cae el golpe dentro del periodo, doblando la cancion sobre si misma."""
    casilla = np.minimum(((indices % periodo) / periodo * casillas).astype(np.int64),
                         casillas - 1)
    suma = np.bincount(casilla, weights=novedad, minlength=casillas)
    cuenta = np.bincount(casilla, minlength=casillas)
    perfil = suma / np.maximum(cuenta, 1)

    # Parabola entre la casilla ganadora y sus vecinas, para no quedarnos con el
    # escalon de 1/casillas de periodo.
    i = int(perfil.argmax())
    y0 = perfil[(i - 1) % casillas]
    y1 = perfil[i]
    y2 = perfil[(i + 1) % casillas]
    divisor = y0 - 2.0 * y1 + y2
    ajuste = 0.5 * (y0 - y2) / divisor if abs(divisor) > 1e-12 else 0.0
    ajuste = max(-0.5, min(0.5, ajuste))
    return ((i + 0.5 + ajuste) / casillas) * periodo


def detectar_tempo(novedad, fps, bpm_min=60.0, bpm_max=200.0, casillas=64):
    """Devuelve (bpm, desfase_en_fotogramas, nitidez) o None.

    La puntuacion compara la energia que hay JUSTO en los pulsos previstos con
    la que hay entre ellos. Es importante que sea asi y no solo "que los golpes
    caigan alineados": doblar la cancion sobre el doble del periodo tambien los
    alinea todos, y con una medida de alineacion a secas un tema de 80 BPM se
    detecta igual de bien como si fuera de 160. Mirando lo que pasa ENTRE los
    pulsos, el error se paga en las dos direcciones:
      - si el tempo va el doble de rapido, la mitad de los pulsos previstos caen
        en silencio y la energia media de los pulsos se hunde;
      - si va la mitad de rapido, los golpes que quedan sin pulso engordan la
        energia de fuera.
    """
    n = len(novedad)
    if n < fps * 4:      # con menos de 4 segundos no hay nada que medir
        return None
    if float(novedad.mean()) <= 1e-12:
        return None

    indices = np.arange(n, dtype=np.float64)
    mejor = None

    for bpm in np.arange(bpm_min, bpm_max + 1e-9, 0.1):
        periodo = fps * 60.0 / bpm
        if periodo < 3.0:
            continue

        desfase = _fase_por_doblado(novedad, indices, periodo, casillas)
        golpes = desfase + np.arange(int((n - 1 - desfase) // periodo) + 1) * periodo
        if len(golpes) < 4:
            continue

        # Un pulso rara vez cae en un fotograma exacto, asi que miramos los dos
        # que lo rodean y nos quedamos con el mayor: la energia esta en uno u otro.
        antes = np.clip(np.floor(golpes).astype(np.int64), 0, n - 1)
        despues = np.clip(antes + 1, 0, n - 1)
        dentro = np.maximum(novedad[antes], novedad[despues])

        mascara = np.ones(n, dtype=bool)
        mascara[antes] = False
        mascara[despues] = False
        if not mascara.any():
            continue
        fuera = float(novedad[mascara].mean())

        nitidez = float(dentro.mean()) / (fuera + 1e-9)
        # Empujoncito flojo hacia los tempos habituales, solo para desempatar.
        preferencia = math.exp(-0.5 * (math.log2(bpm / 120.0) / 1.2) ** 2)
        puntuacion = nitidez * preferencia

        if mejor is None or puntuacion > mejor[0]:
            mejor = (puntuacion, float(bpm), float(desfase), nitidez)

    if mejor is None:
        return None
    _, bpm, desfase, nitidez = mejor
    return bpm, desfase, nitidez


def coherencia_tempo(novedad, fps, bpm_global, trozos_max=6):
    """Cuantos trozos del tema, mirados por separado, dan el mismo tempo.

    Es la medida de confianza que de verdad sirve, y me costo llegar a ella. La
    obvia -comparar la energia que hay en los pulsos con la que hay fuera- da
    numeros altisimos con audio SIN ritmo: si la senal es casi plana, esa
    proporcion sale grande por casualidad. Medido: un pad sin un solo golpe
    sacaba un 97%, mas que dos canciones de verdad.

    Partir el tema y ver si los trozos se ponen de acuerdo no tiene ese problema,
    porque un tempo inventado sale distinto en cada trozo. Medido sobre seis
    audios sin ritmo y trece con el: los primeros dan 0 de 6 sin excepcion, y los
    segundos de 3 a 6.

    Devuelve (cuantos_coinciden, cuantos_trozos). Con (0, 0) el tema es
    demasiado corto para partirlo y no opinamos.
    """
    duracion = len(novedad) / max(fps, 1e-6)
    # Cada trozo necesita unos segundos para que su propia deteccion valga algo.
    if duracion < 20.0:
        return 0, 0
    trozos = int(min(trozos_max, max(2, duracion // 8)))

    n = len(novedad)
    coinciden = 0
    for k in range(trozos):
        i, j = int(n * k / trozos), int(n * (k + 1) / trozos)
        r = detectar_tempo(novedad[i:j], fps)
        if r is None:
            continue
        t = r[0]
        # Vale tambien la mitad o el doble: un trozo sin bombo suele contar los
        # golpes de dos en dos, y eso no es estar en desacuerdo.
        margen = max(1.5, bpm_global * 0.02)
        if min(abs(t - bpm_global), abs(t - bpm_global * 2),
               abs(t - bpm_global / 2)) <= margen:
            coinciden += 1
    return coinciden, trozos


def etiqueta_fiabilidad(v):
    """Los umbrales son orientativos: la prueba de verdad es mirarlo con la musica."""
    if v >= 0.75:
        return "clarisimo"
    if v >= 0.5:
        return "razonable"
    return "dudoso, revisalo"


def valores_de_compas(n_fotogramas, fps, bpm, desfase, pulsos_por_compas, caida_seg):
    """Las cuatro curvas del compas, ya calculadas para todos los fotogramas."""
    periodo = fps * 60.0 / max(bpm, 1e-6)
    t = (np.arange(n_fotogramas, dtype=np.float64) - desfase) / periodo
    indice = np.floor(t)
    resto = t - indice                       # 0..1 dentro del pulso

    # El pulso baja en linea recta desde 1 hasta 0 en `caida_seg` segundos.
    caida_frames = max(caida_seg * fps, 1e-6)
    pulso = np.clip(1.0 - (resto * periodo) / caida_frames, 0.0, 1.0)

    ppc = max(int(pulsos_por_compas), 1)
    numero = np.mod(indice, ppc)
    fase_compas = (numero + resto) / ppc

    return {
        CLAVE_PULSO: pulso,
        CLAVE_FASE_PULSO: resto,
        CLAVE_FASE_COMPAS: fase_compas,
        CLAVE_NUM_PULSO: numero,
    }


def generar_compas(fuente, escena=None):
    """Escribe las propiedades animadas del compas. Devuelve n de pulsos."""
    crudos = leer_crudos(fuente)
    if crudos is None or np is None:
        return -1
    fotogramas, _matriz, _n_bandas = crudos
    aud = fuente.audioviz_audio

    valores = valores_de_compas(len(fotogramas), aud.fps, aud.bpm, aud.desfase_compas,
                                aud.pulsos_por_compas, aud.caida_pulso)

    for clave in CLAVES_COMPAS:
        fuente[clave] = 0.0
        tope = float(max(int(aud.pulsos_por_compas) - 1, 1)) if clave == CLAVE_NUM_PULSO else 1.0
        fuente.id_properties_ui(clave).update(min=0.0, max=tope, soft_min=0.0,
                                              soft_max=tope, default=0.0)

    curvas = crear_curvas(fuente, list(RUTAS_COMPAS))
    for fc, clave in zip(curvas, CLAVES_COMPAS):
        # El numero de pulso va a saltos: interpolarlo seria un sinsentido.
        rellenar_curva(fc, fotogramas, valores[clave].tolist(),
                       interpolacion=0 if clave == CLAVE_NUM_PULSO else 1)

    if escena is not None and aud.marcadores:
        poner_marcadores(escena, fuente)

    periodo = aud.fps * 60.0 / max(aud.bpm, 1e-6)
    return int(max(0, (len(fotogramas) - aud.desfase_compas) // periodo) + 1)


def quitar_compas(fuente, escena=None):
    borrar_curvas(fuente, RUTAS_COMPAS)
    for clave in CLAVES_COMPAS:
        if clave in fuente:
            del fuente[clave]
    if escena is not None:
        quitar_marcadores(escena)


def poner_marcadores(escena, fuente):
    """Un marcador en el primer pulso de cada compas, para verlo en la timeline."""
    quitar_marcadores(escena)
    crudos = leer_crudos(fuente)
    if crudos is None:
        return 0
    fotogramas, _m, _b = crudos
    aud = fuente.audioviz_audio

    periodo = aud.fps * 60.0 / max(aud.bpm, 1e-6)
    paso = periodo * max(int(aud.pulsos_por_compas), 1)
    inicio = fotogramas[0] + aud.desfase_compas
    fin = fotogramas[-1]

    puestos = 0
    k = 0
    while True:
        f = inicio + k * paso
        if f > fin:
            break
        if f >= fotogramas[0]:
            escena.timeline_markers.new(f"{PREFIJO_MARCADOR}{k + 1}", frame=int(round(f)))
            puestos += 1
        k += 1
        if puestos > 2000:      # temas larguisimos: no llenamos la timeline
            break
    return puestos


def quitar_marcadores(escena):
    for m in [x for x in escena.timeline_markers if x.name.startswith(PREFIJO_MARCADOR)]:
        escena.timeline_markers.remove(m)


def _al_cambiar_compas(self, contexto):
    fuente = self.id_data
    if es_fuente(fuente) and self.compas_activo:
        try:
            generar_compas(fuente, contexto.scene)
        except Exception as e:
            print(f"Audio Viz: no he podido generar el compas: {e}")


def _al_cambiar_marcadores(self, contexto):
    fuente = self.id_data
    if not es_fuente(fuente):
        return
    if self.marcadores and self.compas_activo:
        poner_marcadores(contexto.scene, fuente)
    else:
        quitar_marcadores(contexto.scene)


def _al_mover_suavizado(self, contexto):
    """Se dispara sola al arrastrar los deslizadores de ataque/caida.

    `self` son los ajustes pegados a UNA fuente, y self.id_data es ese Empty:
    cada audio de la escena tiene su propio suavizado.
    """
    fuente = self.id_data
    if not es_fuente(fuente):
        return
    try:
        aplicar_suavizado(fuente, self.ataque, self.caida, self.fps)
    except Exception as e:
        print(f"Audio Viz: no he podido aplicar el suavizado: {e}")


def crear_malla_barra(nombre, ancho):
    """Un prisma de base `ancho` y altura 1, apoyado en el origen.

    Que mida exactamente 1 de alto y que el origen este ABAJO es lo que permite
    que el driver de escala en Z sea directamente la altura en metros, y que la
    barra crezca hacia arriba en vez de estirarse por los dos lados.
    """
    m = bpy.data.meshes.new(nombre)
    h = ancho / 2.0
    verts = [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0),
             (-h, -h, 1), (h, -h, 1), (h, h, 1), (-h, h, 1)]
    caras = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    m.from_pydata(verts, [], caras)
    m.validate()
    m.update()
    return m


def crear_material(indice, n_bandas):
    """Material de emision con el tono repartido por el espectro."""
    nombre = f"{PREFIJO_MATERIAL}{indice}"
    mat = bpy.data.materials.get(nombre)
    if mat is None:
        mat = bpy.data.materials.new(nombre)

    # En Blender 5.x los materiales ya vienen con nodos y 'use_nodes' esta en
    # via de desaparicion; solo lo tocamos si hiciera falta (versiones viejas).
    if mat.node_tree is None:
        mat.use_nodes = True

    nodos = mat.node_tree.nodes
    enlaces = mat.node_tree.links
    nodos.clear()

    # Del rojo (graves) al violeta (agudos), como un espectro visible.
    tono = 0.0 + (indice / max(n_bandas - 1, 1)) * 0.78
    r, g, b = colorsys.hsv_to_rgb(tono, 0.85, 1.0)

    emision = nodos.new("ShaderNodeEmission")
    emision.name = "Emision"
    emision.location = (0, 0)
    emision.inputs["Color"].default_value = (r, g, b, 1.0)
    emision.inputs["Strength"].default_value = 1.0

    salida = nodos.new("ShaderNodeOutputMaterial")
    salida.location = (200, 0)
    enlaces.new(emision.outputs["Emission"], salida.inputs["Surface"])
    return mat


def anadir_driver(id_datos, ruta_datos, indice, empty, variables, expresion):
    """Engancha una propiedad a una o varias propiedades del Empty.

    Un driver es una relacion en vivo: "esta propiedad vale lo que diga esa
    otra, pasada por esta formula". No hay claves de animacion en la barra;
    si cambias la altura, todo se recalcula solo.

    `variables` es {nombre_en_la_formula: propiedad_del_empty}, por ejemplo
    {"b": "band_3", "p": "pulso"}: asi una barra puede seguir a su banda Y al
    pulso del compas a la vez.
    """
    fc = id_datos.driver_add(ruta_datos, indice) if indice >= 0 else id_datos.driver_add(ruta_datos)
    d = fc.driver
    d.type = 'SCRIPTED'

    for v in list(d.variables):
        d.variables.remove(v)

    for nombre, propiedad in variables.items():
        var = d.variables.new()
        var.name = nombre
        var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'OBJECT'
        var.targets[0].id = empty
        var.targets[0].data_path = f'["{propiedad}"]'

    # La formula se escribe con los numeros ya metidos para que Blender la
    # reconozca como "expresion simple" y la compile sin pedir permiso para
    # ejecutar Python (el aviso de "Auto Run Python Scripts").
    d.expression = expresion
    return fc


def curva_por_ruta(fuente, ruta):
    """La F-Curve de una propiedad concreta de la fuente, o None."""
    bolsa = obtener_bolsa(fuente, crear=False)
    if bolsa is None:
        return None
    for fc in bolsa.fcurves:
        if fc.data_path == ruta:
            return fc
    return None


def tiene_compas(fuente):
    return (fuente is not None and es_fuente(fuente)
            and fuente.audioviz_audio.compas_activo
            and CLAVE_PULSO in fuente)


# --- drivers de barras y LEDs -----------------------------------------------
# Estos dos presets no se recalculan cada fotograma como el plexus: viven de
# drivers, que son formulas fijas escritas al crearlos. Por eso cada objeto
# guarda a que banda (y a que segmento) pertenece: asi se pueden reescribir sus
# formulas cuando mueves un deslizador, en vez de obligarte a rehacerlo todo.

def canal_util(fuente, canal):
    """Al que se puede recurrir de verdad: si no hay estereo, siempre mono."""
    if canal in ('IZQ', 'DER', 'ESTEREO') and not es_estereo(fuente):
        return 'MONO'
    return canal


def banda_de(fuente, canal, i):
    """Nombre de la propiedad de banda para ese canal."""
    return f"{PREFIJOS_CANAL[canal_util(fuente, canal)]}{i}"


def poner_driver_barra(ob, fuente, aj):
    i = int(ob.get("av_banda", 0))
    # En estereo cada barra guarda a que canal pertenece; en los demas modos
    # todas usan el mismo.
    canal = ob.get("av_canal", "") or aj.barras_canal
    variables = {"b": banda_de(fuente, canal, i)}
    formula = f"{aj.base:.6f} + b * {aj.altura:.6f}"
    if aj.barras_pulso > 0.0 and tiene_compas(fuente):
        variables["p"] = CLAVE_PULSO
        formula += f" + p * {aj.barras_pulso:.6f}"
    anadir_driver(ob, "scale", 2, fuente, variables, formula)


def poner_driver_led(ob, fuente, aj, n_segmentos):
    i = int(ob.get("av_banda", 0))
    j = int(ob.get("av_segmento", 0))
    canal = ob.get("av_canal", "") or aj.led_canal
    umbral = j / max(n_segmentos, 1)
    variables = {"b": banda_de(fuente, canal, i)}
    nivel = "b"
    if aj.led_pulso > 0.0 and tiene_compas(fuente):
        variables["p"] = CLAVE_PULSO
        nivel = f"(b + p * {aj.led_pulso:.6f})"
    anadir_driver(ob, '["av_on"]', -1, fuente, variables,
                  f"min(max(({nivel} - {umbral:.6f}) * {aj.led_dureza:.4f}, 0.0), 1.0)")


def reparto_estereo(bandas, canal):
    """Que columnas dibujar y con que canal cada una.

    En estereo se hace el doble de columnas y en espejo: los graves quedan en el
    centro y los agudos se van a los extremos, con el canal izquierdo a la
    izquierda. Es la disposicion clasica del ecualizador estereo, y ademas asi
    la simetria del dibujo dice de un vistazo si el tema esta centrado.
    """
    if canal != 'ESTEREO':
        return [(i, canal) for i in bandas]
    izquierda = [(i, 'IZQ') for i in reversed(bandas)]   # agudo -> grave
    derecha = [(i, 'DER') for i in bandas]               # grave -> agudo
    return izquierda + derecha


def _al_cambiar_drivers_barras(self, contexto):
    """Reescribe las formulas de las barras que ya existen.

    El padre de cada barra es su fuente de audio, asi que no hace falta buscarla
    por el nombre.
    """
    for ob in contexto.scene.objects:
        if "av_banda" in ob and ob.name.startswith("AV_Barra") and ob.parent is not None:
            try:
                poner_driver_barra(ob, ob.parent, self)
            except Exception as e:
                print(f"Audio Viz: no he podido actualizar {ob.name}: {e}")


def _al_cambiar_drivers_led(self, contexto):
    for ob in contexto.scene.objects:
        if "av_segmento" in ob and ob.name.startswith(PREFIJO_LED) and ob.parent is not None:
            try:
                poner_driver_led(ob, ob.parent, self, self.led_segmentos)
            except Exception as e:
                print(f"Audio Viz: no he podido actualizar {ob.name}: {e}")


# ---------------------------------------------------------------------------
# PRESET 1: ECUALIZADOR DE LEDS
# ---------------------------------------------------------------------------
# Cada banda es una columna de cubos sueltos que se encienden de abajo arriba,
# como el vumetro de LEDs de una cadena de los 80. El segmento j de una columna
# se enciende cuando la banda supera j/total.
#
# El truco para que esto no sean 96 materiales distintos: TODOS los cubos
# comparten una misma malla y un mismo material. Lo que cambia de uno a otro son
# dos propiedades personalizadas del objeto:
#   av_on    : 0 o 1, controlada por un driver (encendido/apagado)
#   av_nivel : 0..1, la altura del segmento en la columna (fija)
# y el material las lee con nodos "Attribute" en modo OBJECT, que devuelven el
# valor de cada objeto por separado aunque el material sea el mismo.

def crear_material_led(etiqueta, brillo, apagado):
    nombre = f"{NOMBRE_MAT_LED}_{etiqueta}"
    mat = bpy.data.materials.get(nombre)
    if mat is None:
        mat = bpy.data.materials.new(nombre)
    if mat.node_tree is None:
        mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()
    enlaces = nt.links

    # --- color segun la ALTURA del segmento: verde -> ambar -> rojo ---
    attr_nivel = nt.nodes.new("ShaderNodeAttribute")
    attr_nivel.attribute_type = 'OBJECT'
    attr_nivel.attribute_name = "av_nivel"
    attr_nivel.location = (-620, 160)

    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.location = (-430, 160)
    rampa.color_ramp.elements[0].position = 0.0
    rampa.color_ramp.elements[0].color = (0.0, 1.0, 0.12, 1.0)   # verde
    rampa.color_ramp.elements[1].position = 1.0
    rampa.color_ramp.elements[1].color = (1.0, 0.03, 0.0, 1.0)   # rojo
    ambar = rampa.color_ramp.elements.new(0.62)
    ambar.color = (1.0, 0.6, 0.0, 1.0)                            # ambar

    # --- brillo segun ENCENDIDO/APAGADO ---
    attr_on = nt.nodes.new("ShaderNodeAttribute")
    attr_on.attribute_type = 'OBJECT'
    attr_on.attribute_name = "av_on"
    attr_on.location = (-620, -140)

    mezcla = nt.nodes.new("ShaderNodeMath")
    mezcla.operation = 'MULTIPLY_ADD'   # resultado = on * (brillo - apagado) + apagado
    mezcla.location = (-430, -140)
    mezcla.inputs[1].default_value = max(brillo - apagado, 0.0)
    mezcla.inputs[2].default_value = apagado

    emision = nt.nodes.new("ShaderNodeEmission")
    emision.name = "Emision"
    emision.location = (-200, 0)
    salida = nt.nodes.new("ShaderNodeOutputMaterial")
    salida.location = (10, 0)

    enlaces.new(attr_nivel.outputs["Factor"], rampa.inputs["Fac"])
    enlaces.new(rampa.outputs["Color"], emision.inputs["Color"])
    enlaces.new(attr_on.outputs["Factor"], mezcla.inputs[0])
    enlaces.new(mezcla.outputs[0], emision.inputs["Strength"])
    enlaces.new(emision.outputs["Emission"], salida.inputs["Surface"])
    return mat


def crear_malla_led(etiqueta, ancho, alto):
    """Una unica malla compartida por todos los segmentos de esta fuente."""
    nombre = f"{PREFIJO_LED}_{etiqueta}_malla"
    vieja = bpy.data.meshes.get(nombre)
    if vieja is not None:
        bpy.data.meshes.remove(vieja)

    m = bpy.data.meshes.new(nombre)
    hx = ancho / 2.0
    hz = alto / 2.0
    verts = [(-hx, -hx, -hz), (hx, -hx, -hz), (hx, hx, -hz), (-hx, hx, -hz),
             (-hx, -hx, hz), (hx, -hx, hz), (hx, hx, hz), (-hx, hx, hz)]
    caras = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    m.from_pydata(verts, [], caras)
    m.validate()
    m.update()
    return m


# ---------------------------------------------------------------------------
# PRESET: EL COMPAS EN CUBOS
# ---------------------------------------------------------------------------
# Un cubo por tiempo del compas, en fila. En cada pulso crece y se enciende el
# que toca, asi que ademas de ver el golpe se ve EN QUE TIEMPO va: el uno, el
# dos, el tres... Es lo que le faltaba a las barras y a los LEDs, donde el
# compas se sumaba a la altura y quedaba disimulado entre las frecuencias.
#
# La formula del driver usa 'numero_pulso', que va a saltos (0, 1, 2, 3):
#     max(0, 1 - abs(n - k))   vale 1 solo cuando n == k
# Es aritmetica pura, asi que Blender la compila como expresion simple y no pide
# permiso para ejecutar Python.

def crear_material_pulso(indice, total, brillo, apagado):
    nombre = f"{NOMBRE_MAT_PULSO}_{indice}"
    mat = bpy.data.materials.get(nombre)
    if mat is None:
        mat = bpy.data.materials.new(nombre)
    if mat.node_tree is None:
        mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()
    # El primer tiempo del compas en un color aparte: es el que marca el "uno".
    if indice == 0:
        color = (1.0, 0.25, 0.15)
    else:
        tono = 0.5 + 0.12 * (indice / max(total - 1, 1))
        color = colorsys.hsv_to_rgb(tono, 0.75, 1.0)

    emision = nt.nodes.new("ShaderNodeEmission")
    emision.name = "Emision"
    emision.location = (-200, 0)
    emision.inputs["Color"].default_value = (*color, 1.0)
    emision.inputs["Strength"].default_value = apagado
    salida = nt.nodes.new("ShaderNodeOutputMaterial")
    salida.location = (10, 0)
    nt.links.new(emision.outputs["Emission"], salida.inputs["Surface"])
    return mat


def _vertices_cubo(lado):
    """Cubo centrado en X e Y y apoyado en Z, para que crezca hacia arriba."""
    h = lado / 2.0
    return [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0),
            (-h, -h, lado), (h, -h, lado), (h, h, lado), (-h, h, lado)]


def crear_malla_cubo(nombre, lado):
    vieja = bpy.data.meshes.get(nombre)
    if vieja is not None:
        bpy.data.meshes.remove(vieja)
    m = bpy.data.meshes.new(nombre)
    caras = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    m.from_pydata(_vertices_cubo(lado), [], caras)
    m.validate()
    m.update()
    return m


def redimensionar_cubo(me, lado):
    """Cambia el tamano sin rehacer la malla, para poder hacerlo en vivo."""
    if len(me.vertices) != 8:
        return
    me.vertices.foreach_set("co", [c for v in _vertices_cubo(lado) for c in v])
    me.update()


def configurar_cubo_pulso(ob, fuente, aj, k, n):
    """Deja un cubo del compas con el tamano, sitio, drivers y brillo que toca.

    La usan tanto el boton de crear como los deslizadores, para que mover un
    control se note al momento en vez de obligar a rehacerlos.
    """
    redimensionar_cubo(ob.data, aj.pulso_lado)
    x0 = -(n - 1) * aj.pulso_separacion / 2.0
    ob.location.x = x0 + k * aj.pulso_separacion

    # 'toca' vale 1 solo en el tiempo k del compas, y 0 en los demas.
    toca = f"max(0.0, 1.0 - abs(n - {k}))"
    variables = {"p": CLAVE_PULSO, "n": CLAVE_NUM_PULSO}
    anadir_driver(ob, "scale", 2, fuente, variables,
                  f"1.0 + p * {toca} * {aj.pulso_crecimiento:.6f}")

    if not ob.data.materials or ob.data.materials[0] is None:
        return
    arbol = ob.data.materials[0].node_tree
    if arbol is None or "Emision" not in arbol.nodes:
        return
    if arbol.animation_data is None:
        arbol.animation_data_create()
    anadir_driver(
        arbol, 'nodes["Emision"].inputs[1].default_value', -1, fuente, variables,
        f"{aj.pulso_apagado:.6f} + p * {toca} * "
        f"{max(aj.pulso_brillo - aj.pulso_apagado, 0.0):.6f}")
    # Rehacer un driver no basta para que se note: hasta que el depsgraph no
    # pasa por ahi sigue viendose el valor anterior.
    arbol.update_tag()
    ob.update_tag()


def _al_cambiar_pulso_vis(self, contexto):
    """Reconfigura los cubos que ya existen al mover cualquier deslizador."""
    cubos = {}
    for ob in contexto.scene.objects:
        if ob.name.startswith(PREFIJO_PULSO_VIS) and "av_pulso_indice" in ob \
                and ob.parent is not None:
            cubos.setdefault(ob.parent.name, []).append(ob)

    for lista in cubos.values():
        lista.sort(key=lambda o: int(o["av_pulso_indice"]))
        for ob in lista:
            try:
                configurar_cubo_pulso(ob, ob.parent, self,
                                      int(ob["av_pulso_indice"]), len(lista))
            except Exception as e:
                print(f"Audio Viz: no he podido actualizar {ob.name}: {e}")


# ---------------------------------------------------------------------------
# PRESET 2: PLEXUS
# ---------------------------------------------------------------------------
# Una nube de puntos que se desplazan segun las bandas y se unen con lineas
# entre los que quedan cerca. Como las lineas cambian al moverse los puntos, la
# malla hay que rehacerla en cada fotograma: eso lo hace un "handler" de Blender
# (una funcion que se dispara sola al cambiar de fotograma, tambien al renderizar).
#
# Las lineas de una malla no se ven en el render por si solas, asi que un
# modificador de Geometry Nodes convierte las aristas en tubos y los puntos en
# esferas. Ese modificador es fijo: lo unico que cambia cada fotograma es la
# malla de puntos y aristas que le entra.

FORMAS = (
    ('ESFERA', "Esfera", "Puntos repartidos por la superficie de una esfera"),
    ('REJILLA', "Rejilla", "Cuadricula plana; los puntos suben y bajan como un terreno"),
    ('NUBE', "Nube", "Puntos sueltos dentro de un volumen esferico"),
    ('ANILLO', "Anillo", "Puntos en circulo que respiran hacia fuera"),
    ('SUPERFICIE', "Piel de un modelo",
     "Puntos repartidos por la superficie de un objeto de la escena; se mueven "
     "hacia fuera siguiendo la normal, como una piel que respira"),
    ('VOLUMEN', "Interior de un modelo",
     "Puntos sueltos por dentro de un objeto de la escena; se mueven en radial "
     "desde su centro"),
)

FORMAS_DE_MODELO = {'SUPERFICIE', 'VOLUMEN'}

ASIGNACIONES = (
    ('RADIAL', "Radial", "La banda depende de la distancia al eje vertical"),
    ('VERTICAL', "Vertical", "La banda depende de la altura: graves abajo, agudos arriba"),
    ('HORIZONTAL', "Horizontal", "La banda depende de la posicion en X, como un espectro tumbado"),
    ('ANGULO', "Angular", "Las bandas dan la vuelta alrededor del centro"),
    ('INDICE', "Alterna", "Punto 1 banda 0, punto 2 banda 1... se entremezclan"),
    ('ALEATORIA', "Aleatoria", "Cada punto recibe una banda al azar"),
)


def generar_puntos(forma, n, radio, semilla):
    """Devuelve (posiciones_base, direcciones_de_movimiento) como arrays (n,3)."""
    rng = np.random.default_rng(semilla)

    if forma == 'ESFERA':
        # Espiral de Fibonacci: el reparto mas uniforme que existe sobre una
        # esfera sin que se apelmacen los polos.
        i = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * i
        d = np.stack([np.cos(theta) * np.sin(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(phi)], axis=1)
        return d * radio, d

    if forma == 'REJILLA':
        lado = int(math.ceil(math.sqrt(max(n, 1))))
        ejes = np.linspace(-1.0, 1.0, lado)
        gx, gy = np.meshgrid(ejes, ejes)
        p = np.stack([gx.ravel(), gy.ravel(), np.zeros(lado * lado)], axis=1)[:n]
        d = np.tile(np.array([0.0, 0.0, 1.0]), (len(p), 1))
        return p * radio, d

    if forma == 'ANILLO':
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        d = np.stack([np.cos(t), np.sin(t), np.zeros(n)], axis=1)
        return d * radio, d

    # NUBE: puntos dentro del volumen, no solo en la superficie.
    v = rng.normal(size=(n, 3))
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    r = rng.random(n) ** (1.0 / 3.0)   # raiz cubica = reparto uniforme en volumen
    return v * r[:, None] * radio, v


# ---------------------------------------------------------------------------
# PUNTOS A PARTIR DE UN MODELO DE LA ESCENA
# ---------------------------------------------------------------------------

def _datos_de_malla(me):
    """Vertices, triangulos y normales de una malla, en arrays de numpy."""
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    nor = np.empty(len(me.loop_triangles) * 3, dtype=np.float64)
    me.loop_triangles.foreach_get("normal", nor)
    return co.reshape(-1, 3), tri.reshape(-1, 3), nor.reshape(-1, 3)


def _matriz_a_local(plexus, origen):
    """Del espacio local del modelo al espacio local del plexus.

    Se calcula UNA vez, al generar los puntos, y queda cocida en ellos. Por eso
    si despues mueves o editas el modelo hay que pulsar 'Regenerar puntos': el
    plexus no persigue al modelo, se limita a nacer encima de el.
    """
    return np.array(plexus.matrix_world.inverted() @ origen.matrix_world, dtype=np.float64)


def _aplicar_matriz(M, puntos):
    return puntos @ M[:3, :3].T + M[:3, 3]


def _aplicar_matriz_normales(M, normales):
    # Las normales no se transforman como los puntos: hay que usar la inversa
    # traspuesta, o con escalados no uniformes dejarian de ser perpendiculares.
    R = np.linalg.inv(M[:3, :3]).T
    n = normales @ R.T
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def puntos_de_superficie(co, tri, nor, n, rng):
    """Reparto uniforme por area: un triangulo grande recibe mas puntos."""
    A, B, C = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        return None, None

    acumulado = np.cumsum(areas) / total
    elegido = np.searchsorted(acumulado, rng.random(n))
    elegido = np.clip(elegido, 0, len(tri) - 1)

    # Coordenadas baricentricas uniformes dentro del triangulo: se sortea un
    # punto del paralelogramo y, si cae en la mitad de fuera, se pliega dentro.
    u = rng.random(n)
    v = rng.random(n)
    fuera = u + v > 1.0
    u[fuera] = 1.0 - u[fuera]
    v[fuera] = 1.0 - v[fuera]

    a, b, c = A[elegido], B[elegido], C[elegido]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a), nor[elegido]


def puntos_de_volumen(co, tri, n, rng):
    """Puntos por dentro del modelo, por sorteo y descarte.

    Para saber si un punto esta dentro se lanza un rayo y se cuentan los cortes
    con la superficie: impar = dentro, par = fuera. Es el metodo de toda la vida
    y funciona con cualquier forma, por retorcida que sea, siempre que la malla
    este cerrada.
    """
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    bvh = BVHTree.FromPolygons([tuple(v) for v in co.tolist()],
                               [tuple(t) for t in tri.tolist()])
    lo, hi = co.min(axis=0), co.max(axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal <= 0.0:
        return None, None
    # Al reanudar el rayo hay que apartarse un pelin de la cara recien cortada,
    # o se vuelve a chocar con ella eternamente.
    epsilon = max(diagonal * 1e-6, 1e-7)
    direccion = Vector((0.0, 0.0, 1.0))

    def dentro(punto):
        origen = Vector(punto)
        cortes = 0
        for _ in range(64):     # tope de seguridad por si la malla esta abierta
            golpe = bvh.ray_cast(origen, direccion)
            if golpe[0] is None:
                break
            cortes += 1
            origen = golpe[0] + direccion * epsilon
        return cortes % 2 == 1

    aceptados = []
    intentos = 0
    tope = n * 60 + 2000
    while len(aceptados) < n and intentos < tope:
        lote = rng.random((256, 3)) * (hi - lo) + lo
        for punto in lote:
            intentos += 1
            if dentro(punto):
                aceptados.append(punto)
                if len(aceptados) >= n:
                    break

    if len(aceptados) < 2:
        return None, None

    P = np.array(aceptados, dtype=np.float64)
    centro = P.mean(axis=0)
    D = P - centro
    largo = np.linalg.norm(D, axis=1, keepdims=True)
    # Un punto que caiga justo en el centro no tiene direccion: le damos una.
    D = np.where(largo > 1e-9, D / np.maximum(largo, 1e-12), np.array([0.0, 0.0, 1.0]))
    return P, D


def puntos_de_modelo(plexus, n, semilla):
    """Muestrea el objeto elegido. Devuelve (posiciones, direcciones) o None."""
    p = plexus.audioviz_plex
    origen = p.objeto_origen
    if origen is None or origen.type != 'MESH':
        return None, None

    rng = np.random.default_rng(semilla)
    dg = bpy.context.evaluated_depsgraph_get()
    evaluado = origen.evaluated_get(dg)     # con los modificadores aplicados
    me = None
    try:
        me = evaluado.to_mesh()
        co, tri, nor = _datos_de_malla(me)
    except Exception as e:
        print(f"Audio Viz: no he podido leer la malla de '{origen.name}': {e}")
        return None, None
    finally:
        if me is not None:
            evaluado.to_mesh_clear()

    if len(tri) == 0:
        return None, None

    M = _matriz_a_local(plexus, origen)

    if p.forma == 'SUPERFICIE':
        P, D = puntos_de_superficie(co, tri, nor, n, rng)
        if P is None:
            return None, None
        return _aplicar_matriz(M, P), _aplicar_matriz_normales(M, D)

    P, D = puntos_de_volumen(co, tri, n, rng)
    if P is None:
        return None, None
    Pl = _aplicar_matriz(M, P)
    Dl = _aplicar_matriz(M, D + P) - Pl      # la direccion es un desplazamiento
    largo = np.linalg.norm(Dl, axis=1, keepdims=True)
    return Pl, Dl / np.maximum(largo, 1e-12)


# ---------------------------------------------------------------------------
# CACHE DE LA DISPOSICION
# ---------------------------------------------------------------------------
# Colocar los puntos cuesta poco en las formas geometricas, pero muestrear el
# volumen de un modelo son decenas de milisegundos. Como la disposicion solo
# depende de la forma, la densidad y la semilla -y NO del fotograma-, se calcula
# una vez y se guarda en el objeto. Cada fotograma solo se recalcula lo que
# cambia: el desplazamiento y las conexiones.

CLAVE_CACHE_P0 = "av_cache_p0"
CLAVE_CACHE_DIR = "av_cache_dir"
CLAVE_CACHE_FIRMA = "av_cache_firma"


def firma_disposicion(p):
    origen = p.objeto_origen.name if p.objeto_origen is not None else ""
    return f"{p.forma}|{p.puntos}|{p.radio:.6f}|{p.semilla}|{origen}"


def disposicion(plexus, forzar=False):
    """(posiciones_base, direcciones), reutilizando el cache si sigue valiendo."""
    p = plexus.audioviz_plex
    firma = firma_disposicion(p)

    if not forzar and plexus.get(CLAVE_CACHE_FIRMA) == firma:
        try:
            p0 = np.array(plexus[CLAVE_CACHE_P0], dtype=np.float64).reshape(-1, 3)
            dirs = np.array(plexus[CLAVE_CACHE_DIR], dtype=np.float64).reshape(-1, 3)
            if len(p0) >= 2 and len(p0) == len(dirs):
                return p0, dirs
        except Exception:
            pass    # cache corrupto: lo rehacemos

    n = max(int(p.puntos), 2)
    p0 = dirs = None
    if p.forma in FORMAS_DE_MODELO:
        p0, dirs = puntos_de_modelo(plexus, n, p.semilla)
    if p0 is None:
        # Sin modelo valido (o modelo abierto/vacio) caemos a la esfera en vez
        # de dejar el plexus en blanco sin explicacion.
        forma = 'ESFERA' if p.forma in FORMAS_DE_MODELO else p.forma
        p0, dirs = generar_puntos(forma, n, p.radio, p.semilla)

    plexus[CLAVE_CACHE_P0] = p0.ravel().tolist()
    plexus[CLAVE_CACHE_DIR] = dirs.ravel().tolist()
    plexus[CLAVE_CACHE_FIRMA] = firma
    return p0, dirs


def tamano_disposicion(plexus):
    """Diagonal de la caja que ocupan los puntos, sin desplazar."""
    p0, _ = disposicion(plexus)
    if len(p0) < 2:
        return 0.0
    return float(np.linalg.norm(p0.max(axis=0) - p0.min(axis=0)))


def distancia_sugerida(plexus, vecinos_objetivo=8):
    """Distancia de union para que cada punto acabe con ~6 vecinos.

    El tamano de un modelo de la escena es imprevisible: puede medir 0.1 o 100.
    Sin esto, elegir un modelo da o una mancha solida o polvo desconectado.

    En vez de multiplicar la separacion media por un factor a ojo, se mide
    directamente a que distancia esta el sexto vecino de cada punto y se coge la
    mediana. Asi se adapta solo: el reparto aleatorio sobre la piel de un modelo
    es mucho mas desigual que una esfera de Fibonacci, y un factor fijo que
    funcione en una deja la otra desconectada.
    """
    p0, _ = disposicion(plexus)
    n = len(p0)
    if n < 2:
        return 0.0
    k = max(1, min(int(vecinos_objetivo), n - 1))

    cuad = (p0 ** 2).sum(axis=1)
    d2 = cuad[:, None] + cuad[None, :] - 2.0 * (p0 @ p0.T)
    np.fill_diagonal(d2, np.inf)
    kesimo = np.partition(d2, k - 1, axis=1)[:, k - 1]
    return float(np.median(np.sqrt(np.maximum(kesimo, 0.0))) * 1.05)


def asignar_bandas(p0, modo, n_bandas, banda_min, banda_max, semilla):
    """Posicion de cada punto DENTRO del espectro, en continuo (p.ej. 3.7).

    Devolver un decimal en vez de un numero de banda entero es lo que permite
    despues interpolar: si todos los puntos de una banda recibieran exactamente
    el mismo valor, una rejilla se partiria en mesetas planas con escalones.
    """
    n = len(p0)
    rng = np.random.default_rng(semilla + 1)

    if modo == 'RADIAL':
        metrica = np.sqrt(p0[:, 0] ** 2 + p0[:, 1] ** 2)
    elif modo == 'VERTICAL':
        metrica = p0[:, 2]
    elif modo == 'HORIZONTAL':
        metrica = p0[:, 0]
    elif modo == 'ANGULO':
        metrica = np.arctan2(p0[:, 1], p0[:, 0])
    elif modo == 'ALEATORIA':
        metrica = rng.random(n)
    else:
        metrica = np.arange(n, dtype=float)

    # Algunas combinaciones no tienen sentido geometrico: en un anillo todos los
    # puntos estan a la misma distancia del centro, asi que 'Radial' los pondria
    # todos en la misma banda. Si la metrica no varia, alternamos.
    ancho = float(metrica.max() - metrica.min())
    if ancho < 1e-6:
        metrica = np.arange(n, dtype=float)
        ancho = float(max(metrica.max() - metrica.min(), 1.0))

    b0 = max(0, min(banda_min, n_bandas - 1))
    b1 = max(b0, min(banda_max, n_bandas - 1))
    cuantas = b1 - b0 + 1

    if modo == 'INDICE':
        # Este modo es discreto por definicion: punto 1 banda 0, punto 2 banda 1...
        return ((np.arange(n) % cuantas) + b0).astype(float)

    t = (metrica - metrica.min()) / ancho
    return b0 + t * (cuantas - 1)


def obtener_nodos_plexus():
    """El grupo de nodos, UNO SOLO compartido por todos los plexus de la escena.

    El grosor, el tamano del punto y el material no estan cocidos dentro: son
    entradas del grupo, y cada objeto les da su propio valor desde su
    modificador. Por eso puede haber diez plexus con diez aspectos distintos
    sin diez grupos de nodos duplicados.
    """
    ng = bpy.data.node_groups.get(NOMBRE_GN_PLEXUS)
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(NOMBRE_GN_PLEXUS, "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Grosor", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket(name="Punto", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket(name="Material", in_out='INPUT', socket_type='NodeSocketMaterial')

    nodos = ng.nodes
    enlaces = ng.links
    entrada = nodos.new("NodeGroupInput"); entrada.location = (-620, 0)
    salida = nodos.new("NodeGroupOutput"); salida.location = (520, 0)

    # --- rama 1: aristas -> tubos ---
    a_curva = nodos.new("GeometryNodeMeshToCurve"); a_curva.location = (-390, 170)
    perfil = nodos.new("GeometryNodeCurvePrimitiveCircle"); perfil.location = (-390, -10)
    perfil.inputs["Resolution"].default_value = 6   # hexagono: de sobra para un hilo
    a_malla = nodos.new("GeometryNodeCurveToMesh"); a_malla.location = (-160, 170)
    mat_tubos = nodos.new("GeometryNodeSetMaterial"); mat_tubos.location = (70, 170)

    enlaces.new(entrada.outputs[0], a_curva.inputs["Mesh"])
    enlaces.new(a_curva.outputs["Curve"], a_malla.inputs["Curve"])
    enlaces.new(entrada.outputs["Grosor"], perfil.inputs["Radius"])
    enlaces.new(perfil.outputs["Curve"], a_malla.inputs["Profile Curve"])
    enlaces.new(a_malla.outputs["Mesh"], mat_tubos.inputs["Geometry"])
    enlaces.new(entrada.outputs["Material"], mat_tubos.inputs["Material"])

    # --- rama 2: puntos -> esferas ---
    bola = nodos.new("GeometryNodeMeshIcoSphere"); bola.location = (-390, -230)
    bola.inputs["Subdivisions"].default_value = 2
    mat_bolas = nodos.new("GeometryNodeSetMaterial"); mat_bolas.location = (-160, -230)
    instanciar = nodos.new("GeometryNodeInstanceOnPoints"); instanciar.location = (70, -230)

    enlaces.new(entrada.outputs["Punto"], bola.inputs["Radius"])
    enlaces.new(bola.outputs["Mesh"], mat_bolas.inputs["Geometry"])
    enlaces.new(entrada.outputs["Material"], mat_bolas.inputs["Material"])
    enlaces.new(entrada.outputs[0], instanciar.inputs["Points"])
    enlaces.new(mat_bolas.outputs["Geometry"], instanciar.inputs["Instance"])

    unir = nodos.new("GeometryNodeJoinGeometry"); unir.location = (300, 0)
    enlaces.new(mat_tubos.outputs["Geometry"], unir.inputs["Geometry"])
    enlaces.new(instanciar.outputs["Instances"], unir.inputs["Geometry"])
    enlaces.new(unir.outputs["Geometry"], salida.inputs[0])
    return ng


def identificador_entrada(ng, nombre):
    """El nombre interno ('Socket_2') de una entrada del grupo, que es como hay
    que dirigirse a ella desde el modificador."""
    for item in ng.interface.items_tree:
        if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == nombre:
            return item.identifier
    return None


def modificador_plexus(ob):
    for m in ob.modifiers:
        if m.type == 'NODES' and m.node_group is not None \
                and m.node_group.name == NOMBRE_GN_PLEXUS:
            return m
    return None


def aplicar_estilo_plexus(ob):
    """Vuelca grosor y tamano de punto de ESTE objeto a su modificador."""
    m = modificador_plexus(ob)
    if m is None:
        return
    p = ob.audioviz_plex
    for nombre, valor in (("Grosor", p.grosor), ("Punto", p.tam_punto)):
        ident = identificador_entrada(m.node_group, nombre)
        if ident is not None:
            m[ident] = valor
    ob.update_tag()


def material_de_plexus(ob, crear=False):
    m = modificador_plexus(ob)
    if m is None:
        return None
    ident = identificador_entrada(m.node_group, "Material")
    if ident is None:
        return None
    mat = m[ident]
    if mat is None and crear:
        mat = bpy.data.materials.new(NOMBRE_MAT_PLEXUS)
        m[ident] = mat
    return mat


def actualizar_material_plexus(ob):
    """Reconstruye el shader de ESTE plexus con sus tres colores y su brillo."""
    mat = material_de_plexus(ob, crear=True)
    if mat is None:
        return
    if mat.node_tree is None:
        mat.use_nodes = True
    p = ob.audioviz_plex

    nt = mat.node_tree
    nt.nodes.clear()

    # El color sale de la banda que mueve cada punto, guardada como atributo de
    # la malla. Al convertir aristas en tubos el atributo viaja con ellas, asi
    # que cada linea se degrada entre las dos bandas que une.
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = "av_nivel"
    attr.location = (-620, 0)

    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.location = (-430, 0)
    rampa.color_ramp.elements[0].position = 0.0
    rampa.color_ramp.elements[0].color = (*p.color_grave, 1.0)
    rampa.color_ramp.elements[1].position = 1.0
    rampa.color_ramp.elements[1].color = (*p.color_agudo, 1.0)
    medio = rampa.color_ramp.elements.new(0.5)
    medio.color = (*p.color_medio, 1.0)

    emision = nt.nodes.new("ShaderNodeEmission")
    emision.name = "Emision"
    emision.location = (-200, 0)
    emision.inputs["Strength"].default_value = p.brillo
    salida = nt.nodes.new("ShaderNodeOutputMaterial")
    salida.location = (10, 0)

    nt.links.new(attr.outputs["Factor"], rampa.inputs["Fac"])
    nt.links.new(rampa.outputs["Color"], emision.inputs["Color"])
    nt.links.new(emision.outputs["Emission"], salida.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------
# CARAS DEL PLEXUS (objeto aparte)
# ---------------------------------------------------------------------------
# Donde tres puntos estan unidos entre si de dos en dos hay un triangulo. Esas
# caras van a un OBJETO SEPARADO, no al mismo: asi puedes darles un material
# traslucido de membrana sin tocar el de las lineas, ocultarlas en el viewport
# y dejarlas solo en el render, o mandarlas a otra capa. El objeto de caras es
# hijo del plexus, asi que lo sigue si lo mueves.

def buscar_triangulos(pares, n):
    """Triangulos cerrados dentro de la lista de aristas."""
    if not len(pares):
        return []
    lista = pares.tolist()
    adyacentes = [set() for _ in range(n)]
    for a, b in lista:
        adyacentes[a].add(b)
        adyacentes[b].add(a)

    tris = []
    for a, b in lista:
        # Los vecinos comunes de a y b cierran triangulo. Con c > b cada
        # triangulo sale una sola vez y no tres.
        for c in adyacentes[a] & adyacentes[b]:
            if c > b:
                tris.append((a, b, c))
    return tris


def filtrar_triangulos(tris, ratio, semilla):
    """Deja pasar solo una fraccion de los triangulos.

    La decision NO puede ser un sorteo nuevo en cada fotograma: las caras
    parpadearian como un fluorescente estropeado. Se calcula un numero
    pseudoaleatorio a partir de los TRES indices de vertice del triangulo, que
    son estables a lo largo del tiempo, asi que un triangulo concreto siempre
    recibe la misma respuesta mientras exista.
    """
    if ratio >= 1.0 or not tris:
        return tris
    if ratio <= 0.0:
        return []

    a = np.array(tris, dtype=np.uint64)
    # Constantes primas grandes, la mezcla de toda la vida para hashear indices.
    h = a[:, 0] * np.uint64(73856093)
    h ^= a[:, 1] * np.uint64(19349663)
    h ^= a[:, 2] * np.uint64(83492791)
    h ^= np.uint64((int(semilla) * 2654435761) & 0xFFFFFFFFFFFFFFFF)

    # Avalancha (el finalizador de splitmix64). Sin esto la semilla apenas hacia
    # nada: mezclarla con un XOR dejaba casi todos los bits del hash intactos y
    # muy pocos triangulos cruzaban el umbral. Aqui cada bit de entrada acaba
    # afectando a todos los de salida. Va en uint64 porque necesitamos que la
    # multiplicacion desborde y de la vuelta, que es justo lo que mezcla.
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)

    valor = (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    return a[valor < ratio].astype(np.int64).tolist()


def crear_material_caras(plexus):
    p = plexus.audioviz_plex
    mat = bpy.data.materials.new(NOMBRE_MAT_CARAS)
    if mat.node_tree is None:
        mat.use_nodes = True
    # En EEVEE de 5.0 la transparencia se elige aqui; 'BLENDED' es la mezcla
    # de verdad (el modo por defecto solo simula con tramado).
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'BLENDED'
    mat.use_backface_culling = False

    nt = mat.node_tree
    nt.nodes.clear()

    # El color sale del atributo av_intensidad: cuanto suena la banda de cada
    # vertice AHORA. Al ser de dominio punto, el shader lo interpola y cada cara
    # queda degradada entre sus tres esquinas.
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.name = "Atributo"
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = "av_intensidad"
    attr.location = (-860, 60)

    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.name = "Rampa"
    rampa.location = (-670, 60)

    emision = nt.nodes.new("ShaderNodeEmission")
    emision.name = "Emision"
    emision.location = (-380, 60)
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (-380, -120)
    mezcla = nt.nodes.new("ShaderNodeMixShader")
    mezcla.name = "Mezcla"
    mezcla.location = (-150, 0)
    salida = nt.nodes.new("ShaderNodeOutputMaterial")
    salida.location = (60, 0)

    nt.links.new(attr.outputs["Factor"], rampa.inputs["Fac"])
    nt.links.new(rampa.outputs["Color"], emision.inputs["Color"])
    nt.links.new(transp.outputs[0], mezcla.inputs[1])
    nt.links.new(emision.outputs[0], mezcla.inputs[2])
    nt.links.new(mezcla.outputs[0], salida.inputs["Surface"])
    _volcar_ajustes_caras(mat, p)
    return mat


def _volcar_ajustes_caras(mat, p):
    nt = mat.node_tree
    if nt is None or not {"Emision", "Mezcla", "Rampa", "Atributo"} <= set(nt.nodes.keys()):
        return
    nt.nodes["Atributo"].attribute_name = p.atributo_caras
    nt.nodes["Emision"].inputs["Strength"].default_value = p.brillo_caras
    nt.nodes["Mezcla"].inputs[0].default_value = p.opacidad_caras

    rampa = nt.nodes["Rampa"].color_ramp
    # Si el degradado esta apagado, los dos extremos comparten color y la cara
    # queda plana; el nodo sigue ahi por si quieres retocarlo a mano.
    alto = p.color_caras_alta if p.degradado_caras else p.color_caras
    rampa.elements[0].position = 0.0
    rampa.elements[0].color = (*p.color_caras, 1.0)
    rampa.elements[1].position = 1.0
    rampa.elements[1].color = (*alto, 1.0)


def objeto_caras_de(plexus, crear=False):
    p = plexus.audioviz_plex
    ob = p.objeto_caras
    if ob is not None and ob.name in bpy.data.objects:
        return ob
    if not crear:
        return None

    nombre = plexus.name + SUFIJO_CARAS
    me = bpy.data.meshes.new(nombre)
    ob = bpy.data.objects.new(nombre, me)
    destino = plexus.users_collection[0] if plexus.users_collection else None
    (destino or bpy.context.scene.collection).objects.link(ob)

    # Hijo del plexus SIN compensar la transformacion: los vertices que le
    # escribimos estan en el espacio local del plexus y tienen que caer encima.
    ob.parent = plexus
    ob.matrix_parent_inverse.identity()
    ob.location = (0.0, 0.0, 0.0)

    me.materials.append(crear_material_caras(plexus))
    p.objeto_caras = ob
    return ob


def actualizar_material_caras(plexus):
    ob = objeto_caras_de(plexus)
    if ob is None or not ob.data.materials:
        return
    _volcar_ajustes_caras(ob.data.materials[0], plexus.audioviz_plex)


def quitar_objeto_caras(plexus):
    p = plexus.audioviz_plex
    ob = p.objeto_caras
    p.objeto_caras = None
    if ob is None or ob.name not in bpy.data.objects:
        return
    malla = ob.data
    mats = [m for m in (malla.materials if malla else []) if m is not None]
    bpy.data.objects.remove(ob, do_unlink=True)
    if malla is not None and malla.users == 0:
        bpy.data.meshes.remove(malla)
    for m in mats:
        if m.users == 0:
            bpy.data.materials.remove(m)


def poner_atributo(me, nombre, valores):
    """Escribe (creandolo si hace falta) un atributo de punto en la malla."""
    attr = me.attributes.get(nombre)
    if attr is None:
        attr = me.attributes.new(nombre, 'FLOAT', 'POINT')
    attr.data.foreach_set("value", valores)
    return attr


def escribir_caras(plexus, puntos, pares, niveles, intensidades):
    ob = objeto_caras_de(plexus)
    if ob is None:
        return 0
    p = plexus.audioviz_plex
    tris = filtrar_triangulos(buscar_triangulos(pares, len(puntos)),
                              p.ratio_caras, p.semilla)

    me = ob.data
    me.clear_geometry()
    me.from_pydata(puntos.tolist(), [], tris)
    # Los mismos atributos que el plexus: al ser de dominio PUNTO, el shader los
    # interpola por la cara y cada triangulo recibe un degradado entre los
    # valores de sus tres vertices.
    poner_atributo(me, "av_nivel", niveles)
    poner_atributo(me, "av_intensidad", intensidades)
    me.update()
    return len(tris)


# ---------------------------------------------------------------------------
# PRESET 3: PAISAJE QUE AVANZA
# ---------------------------------------------------------------------------
# Una rejilla donde un eje son las frecuencias y el OTRO ES EL TIEMPO: cada fila
# es un instante distinto del pasado y la altura es lo que sonaba entonces. Como
# en cada fotograma todas las filas se corren un paso, el relieve parece avanzar
# hacia el horizonte. Es el clasico "paisaje de espectro".
#
# Esto es lo que lo hace distinto del plexus y lo que justifica un preset aparte:
#   - el plexus lee el fotograma ACTUAL; esto lee el HISTORICO;
#   - el plexus busca vecinos por distancia en cada fotograma; aqui la rejilla es
#     fija y solo cambian las alturas, asi que sale mucho mas barato: una malla
#     de 128x128 (16.000 vertices) se actualiza en menos de 2 ms.
#
# Las alturas se leen de las CURVAS de la fuente, no de los valores en crudo, y
# por eso el paisaje respeta el ataque y la caida que tengas puestos.

DIRECCIONES = (
    ('SUR', "Hacia el sur (-Y)", "El relieve viene de lejos y se acerca al observador"),
    ('NORTE', "Hacia el norte (+Y)", "El relieve se aleja del observador"),
    ('OESTE', "Hacia el oeste (-X)", "Avanza hacia la izquierda"),
    ('ESTE', "Hacia el este (+X)", "Avanza hacia la derecha"),
)

MODOS_PAISAJE = (
    ('SOLIDO', "Solido", "Solo la superficie"),
    ('MALLA', "Malla", "Solo los hilos de la rejilla, en plan retro"),
    ('AMBOS', "Solido + malla", "La superficie con la rejilla marcada encima"),
)


def es_paisaje(ob):
    return (ob is not None and ob.type == 'MESH'
            and getattr(ob, "audioviz_paisaje", None) is not None
            and ob.audioviz_paisaje.es_paisaje)


def paisajes_de_la_escena(escena):
    return [o for o in escena.objects if es_paisaje(o)]


def firma_paisaje(p):
    return (f"{p.filas}|{p.columnas}|{p.repeticiones}|{int(p.espejo)}|"
            f"{p.ancho:.4f}|{p.largo:.4f}|{p.direccion}")


def columnas_totales(p):
    """Las copias comparten la columna de la costura, para no duplicar vertices."""
    return p.columnas * p.repeticiones - (p.repeticiones - 1)


def recorrido_espectro(p):
    """Posicion dentro del espectro (0 grave, 1 agudo) de cada columna.

    Con varias repeticiones el barrido de frecuencias se repite a lo ancho. En
    modo espejo las copias van alternas -grave...agudo, agudo...grave- para que
    en la union coincidan y no se vea la costura.
    """
    total = columnas_totales(p)
    t = np.linspace(0.0, float(p.repeticiones), total)
    if p.espejo:
        # Triangulo: 0 -> 1 -> 0 -> 1 ...
        return np.abs(((t + 1.0) % 2.0) - 1.0)
    return np.mod(t, 1.0)


def rejilla_paisaje(p):
    """Coordenadas XY de la rejilla. Fila 0 = lo que suena ahora."""
    total = columnas_totales(p)
    u = np.linspace(0.0, 1.0, total)          # a lo ancho, ya con las repeticiones
    v = np.linspace(0.0, 1.0, p.filas)        # a lo largo: 0 = ahora, 1 = lo mas viejo
    U, V = np.meshgrid(u, v)

    ancho = p.ancho * p.repeticiones          # cada copia mide 'ancho'
    largo = p.largo
    if p.direccion == 'SUR':
        x, y = (U - 0.5) * ancho, (0.5 - V) * largo
    elif p.direccion == 'NORTE':
        x, y = (U - 0.5) * ancho, (V - 0.5) * largo
    elif p.direccion == 'OESTE':
        x, y = (0.5 - V) * largo, (U - 0.5) * ancho
    else:                                      # ESTE
        x, y = (V - 0.5) * largo, (U - 0.5) * ancho

    co = np.zeros((p.filas * total, 3), dtype=np.float64)
    co[:, 0] = x.ravel()
    co[:, 1] = y.ravel()
    return co, U, V


def construir_malla_paisaje(ob):
    """Rehace la topologia solo si ha cambiado: es lo unico caro que hay aqui."""
    p = ob.audioviz_paisaje
    firma = firma_paisaje(p)
    me = ob.data
    total = columnas_totales(p)
    if ob.get(CLAVE_FIRMA_PAISAJE) == firma and len(me.vertices) == p.filas * total:
        return False

    co, U, V = rejilla_paisaje(p)
    filas = p.filas
    j, i = np.meshgrid(np.arange(filas - 1), np.arange(total - 1), indexing='ij')
    esquina = (j * total + i).ravel()
    caras = np.stack([esquina, esquina + 1, esquina + total + 1, esquina + total], axis=1)

    me.clear_geometry()
    me.from_pydata(co.tolist(), [], caras.tolist())
    me.update()

    # av_nivel (donde cae en el espectro) y av_tiempo (como de viejo) son fijos:
    # dependen de la rejilla, no de la musica. Se escriben una sola vez.
    espectro = recorrido_espectro(p)
    poner_atributo(me, "av_nivel",
                   np.tile(espectro, filas).astype(np.float32))
    poner_atributo(me, "av_tiempo", V.ravel().astype(np.float32))

    ob[CLAVE_FIRMA_PAISAJE] = firma
    return True


def actualizar_paisaje(escena, ob):
    """Recalcula las alturas del paisaje en el fotograma actual."""
    if not es_paisaje(ob) or np is None:
        return -1
    p = ob.audioviz_paisaje
    fuente = p.fuente if es_fuente(p.fuente) else fuente_activa(escena)
    if fuente is None:
        return -1

    n_bandas = int(fuente.get(CLAVE_BANDAS, 8))
    curvas = curvas_de_bandas(fuente, n_bandas)
    marcos = fuente.get(CLAVE_FRAMES)
    if curvas is None or not marcos:
        return -1
    primero, ultimo = int(marcos[0]), int(marcos[-1])

    construir_malla_paisaje(ob)
    co, U, V = rejilla_paisaje(p)
    filas = p.filas

    # Fila 0 = ahora; cada fila mira mas atras en el tiempo. Con pasos decimales
    # el relieve avanza a fracciones de fila y el movimiento sale continuo.
    tiempos = escena.frame_current - np.arange(filas) * p.fotogramas_por_fila
    canal = canal_util(fuente, p.canal)
    if canal == 'ESTEREO':
        juegos = [curvas_de_bandas(fuente, n_bandas, 'IZQ'),
                  curvas_de_bandas(fuente, n_bandas, 'DER')]
        juegos = [c if c is not None else curvas for c in juegos]
    else:
        juegos = [curvas_de_bandas(fuente, n_bandas, canal) or curvas]

    lecturas = []
    for juego in juegos:
        valores = np.zeros((filas, n_bandas), dtype=np.float64)
        for j, t in enumerate(tiempos):
            if t < primero or t > ultimo:
                continue                  # fuera del audio: llano
            for b in range(n_bandas):
                valores[j, b] = juego[b].evaluate(float(t))
        lecturas.append(valores)

    # De las bandas a todas las columnas, repeticiones incluidas.
    b0 = max(0, min(p.banda_min, n_bandas - 1))
    b1 = max(b0, min(p.banda_max, n_bandas - 1))
    espectro = recorrido_espectro(p)            # 0..1 dentro del rango elegido
    posiciones = b0 + espectro * (b1 - b0)

    def a_columnas(valores):
        if p.suave and b1 > b0:
            bajo = np.floor(posiciones).astype(np.int64)
            alto = np.clip(bajo + 1, 0, n_bandas - 1)
            peso = posiciones - bajo
            return valores[:, bajo] * (1.0 - peso) + valores[:, alto] * peso
        cercano = np.clip(np.round(posiciones).astype(np.int64), 0, n_bandas - 1)
        return valores[:, cercano]

    alturas = a_columnas(lecturas[0])
    if len(lecturas) > 1:
        # Estereo: el terreno se reparte a lo ANCHO de la rejilla, que es el eje
        # que ocupa el espectro. Se usa la posicion de la columna y no la X del
        # mundo para que siga funcionando igual gire hacia donde gire el paisaje.
        u = np.linspace(0.0, 1.0, len(posiciones))
        peso = u * u * (3.0 - 2.0 * u)          # suave por el centro, plano en los bordes
        alturas = alturas * (1.0 - peso)[None, :] + a_columnas(lecturas[1]) * peso[None, :]

    alturas = np.clip(alturas, 0.0, 1.0)

    # --- moldear el relieve ---
    # La curva se aplica primero, sobre el 0..1 limpio: asi hace siempre lo
    # mismo, suba o baje la ganancia despues.
    if abs(p.curva - 1.0) > 1e-6:
        alturas = np.power(alturas, max(p.curva, 1e-4))

    if abs(p.inclinacion) > 1e-6:
        # Balanza entre graves y agudos, como un ecualizador de inclinacion.
        peso_banda = 1.0 + p.inclinacion * (2.0 * espectro - 1.0)
        alturas = alturas * peso_banda[None, :]

    if abs(p.ganancia - 1.0) > 1e-6:
        alturas = alturas * p.ganancia

    if p.suelo > 0.0:
        alturas = p.suelo + (1.0 - p.suelo) * alturas

    # El pulso se muestrea POR FILA, no una vez para todo el paisaje: cada fila
    # es un instante distinto, asi que cada golpe levanta la fila que le toca y
    # se convierte en una cresta que viaja con el terreno.
    if tiene_compas(fuente) and (p.pulso_altura != 0.0 or p.compas_marca != 0.0):
        fc_pulso = curva_por_ruta(fuente, f'["{CLAVE_PULSO}"]')
        dentro = (tiempos >= primero) & (tiempos <= ultimo)
        pulsos = np.zeros(filas)
        if fc_pulso is not None:
            pulsos = np.array([fc_pulso.evaluate(float(t)) if d else 0.0
                               for t, d in zip(tiempos, dentro)])

        if p.pulso_altura != 0.0:
            if p.pulso_extension >= 1.0:
                # Cresta recta de lado a lado.
                alturas = alturas + (pulsos * p.pulso_altura)[:, None]
            else:
                # Campana centrada en los graves: la cresta nace donde esta el
                # bombo y se apaga hacia los agudos, en vez de ser un escalon
                # recto que delata la rejilla.
                perfil = np.exp(-(espectro / max(p.pulso_extension, 1e-3)) ** 2)
                alturas = alturas + np.outer(pulsos * p.pulso_altura, perfil)

        # Marca de compas: solo en el PRIMER tiempo y solo en una orilla. Con la
        # cresta sola todos los tiempos son iguales y no se ve donde empieza cada
        # compas; esto deja un carril de muescas al borde que viaja con el
        # terreno y funciona como una regla.
        #
        # Por defecto va en los AGUDOS porque la cresta del pulso se concentra en
        # los graves: puestas en el mismo sitio se taparian.
        if p.compas_marca != 0.0:
            fc_num = curva_por_ruta(fuente, f'["{CLAVE_NUM_PULSO}"]')
            if fc_num is not None:
                numeros = np.array([fc_num.evaluate(float(t)) if d else 1.0
                                    for t, d in zip(tiempos, dentro)])
                fuerte = (numeros < 0.5).astype(np.float64)
                ancho = max(p.compas_marca_ancho, 1e-3)
                # Meseta pegada al borde con la caida suave, no una cuña.
                if p.compas_marca_lado == 'AGUDOS':
                    orilla = np.clip((espectro - (1.0 - ancho)) / (ancho * 0.35), 0.0, 1.0)
                else:
                    orilla = np.clip((ancho - espectro) / (ancho * 0.35), 0.0, 1.0)
                alturas = alturas + np.outer(pulsos * fuerte * p.compas_marca, orilla)

    alturas = np.clip(alturas, 0.0, 1.0)
    co[:, 2] = (alturas * p.altura).ravel()

    me = ob.data
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    poner_atributo(me, "av_intensidad", alturas.ravel().astype(np.float32))
    me.update()
    return int(alturas.size)


def aplicar_estilo_paisaje(ob):
    """El aspecto vive entero en el material, asi que basta con rehacerlo.

    La primera version usaba el modificador Wireframe de Blender para los hilos.
    Funcionaba, pero al medirlo resulto que costaba 20 ms por fotograma frente a
    1 ms del resto: regeneraba 28.000 vertices de tubos cada vez. Dibujar la
    rejilla en el shader no genera geometria y ademas sale perfectamente
    alineada, porque se calcula con av_nivel y av_tiempo, que son justo las
    coordenadas de la rejilla.
    """
    for m in [x for x in ob.modifiers if x.type == 'WIREFRAME']:
        ob.modifiers.remove(m)      # limpiamos el de versiones anteriores
    actualizar_material_paisaje(ob)


def _rejilla_en_shader(nt, attr_nivel, attr_tiempo, p, x, y):
    """Devuelve el socket con 1 en las lineas de la rejilla y 0 entre ellas.

    'PINGPONG' con escala 0.5 da directamente la distancia al entero mas
    cercano: 0 justo en la linea, 0.5 en el centro de la celda.
    """
    salidas = []
    for attr, divisiones, dy in ((attr_nivel, p.columnas - 1, y),
                                 (attr_tiempo, p.filas - 1, y - 160)):
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = 'MULTIPLY'
        mul.location = (x, dy)
        mul.inputs[1].default_value = float(max(divisiones, 1))

        pong = nt.nodes.new("ShaderNodeMath")
        pong.operation = 'PINGPONG'
        pong.location = (x + 180, dy)
        pong.inputs[1].default_value = 0.5

        nt.links.new(attr.outputs["Factor"], mul.inputs[0])
        nt.links.new(mul.outputs[0], pong.inputs[0])
        salidas.append(pong.outputs[0])

    cerca = nt.nodes.new("ShaderNodeMath")
    cerca.operation = 'MINIMUM'
    cerca.location = (x + 360, y - 80)
    nt.links.new(salidas[0], cerca.inputs[0])
    nt.links.new(salidas[1], cerca.inputs[1])

    linea = nt.nodes.new("ShaderNodeMapRange")
    linea.location = (x + 540, y - 80)
    linea.inputs["From Min"].default_value = 0.0
    linea.inputs["From Max"].default_value = max(p.grosor_malla, 1e-4)
    linea.inputs["To Min"].default_value = 1.0
    linea.inputs["To Max"].default_value = 0.0
    linea.clamp = True
    linea.interpolation_type = 'SMOOTHSTEP'
    nt.links.new(cerca.outputs[0], linea.inputs["Value"])
    return linea.outputs["Result"]


def material_de_paisaje(ob, crear=False):
    me = ob.data
    if me.materials and me.materials[0] is not None:
        return me.materials[0]
    if not crear:
        return None
    mat = bpy.data.materials.new(NOMBRE_MAT_PAISAJE)
    me.materials.append(mat)
    return mat


def actualizar_material_paisaje(ob):
    """Monta el shader: color por altura, rejilla dibujada y horizonte difuminado."""
    mat = material_de_paisaje(ob, crear=True)
    if mat is None:
        return
    if mat.node_tree is None:
        mat.use_nodes = True
    p = ob.audioviz_paisaje

    transparencias = (p.modo != 'SOLIDO') or p.desvanecer
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'BLENDED' if transparencias else 'DITHERED'
    mat.use_backface_culling = False

    nt = mat.node_tree
    nt.nodes.clear()

    # --- color segun la altura ---
    attr_int = nt.nodes.new("ShaderNodeAttribute")
    attr_int.attribute_type = 'GEOMETRY'
    attr_int.attribute_name = "av_intensidad"
    attr_int.location = (-1500, 320)

    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.location = (-1320, 320)
    rampa.color_ramp.elements[0].position = 0.0
    rampa.color_ramp.elements[0].color = (*p.color_bajo, 1.0)
    rampa.color_ramp.elements[1].position = 1.0
    rampa.color_ramp.elements[1].color = (*p.color_alto, 1.0)

    emision = nt.nodes.new("ShaderNodeEmission")
    emision.name = "Emision"
    emision.location = (-380, 320)
    emision.inputs["Strength"].default_value = p.brillo
    nt.links.new(attr_int.outputs["Factor"], rampa.inputs["Fac"])
    nt.links.new(rampa.outputs["Color"], emision.inputs["Color"])

    salida = nt.nodes.new("ShaderNodeOutputMaterial")
    salida.location = (120, 0)

    attr_nivel = nt.nodes.new("ShaderNodeAttribute")
    attr_nivel.attribute_type = 'GEOMETRY'
    attr_nivel.attribute_name = "av_nivel"
    attr_nivel.location = (-1500, 40)
    attr_tiempo = nt.nodes.new("ShaderNodeAttribute")
    attr_tiempo.attribute_type = 'GEOMETRY'
    attr_tiempo.attribute_name = "av_tiempo"
    attr_tiempo.location = (-1500, -160)

    # --- cuanto se ve en cada punto ---
    if p.modo == 'SOLIDO':
        opacidad = None
    else:
        linea = _rejilla_en_shader(nt, attr_nivel, attr_tiempo, p, -1320, 40)
        if p.modo == 'MALLA':
            opacidad = linea
        else:
            # La superficie queda como un velo y los hilos encima.
            maximo = nt.nodes.new("ShaderNodeMath")
            maximo.operation = 'MAXIMUM'
            maximo.location = (-620, -40)
            maximo.inputs[1].default_value = p.opacidad_superficie
            nt.links.new(linea, maximo.inputs[0])
            opacidad = maximo.outputs[0]

    if p.desvanecer:
        # Sin esto el relieve se corta en seco con una linea recta que delata
        # que aquello es una rejilla y no un horizonte.
        fundido = nt.nodes.new("ShaderNodeMapRange")
        fundido.location = (-620, -260)
        fundido.inputs["From Min"].default_value = max(0.0, 1.0 - p.desvanecido)
        fundido.inputs["From Max"].default_value = 1.0
        fundido.inputs["To Min"].default_value = 1.0
        fundido.inputs["To Max"].default_value = 0.0
        fundido.clamp = True
        nt.links.new(attr_tiempo.outputs["Factor"], fundido.inputs["Value"])

        if opacidad is None:
            opacidad = fundido.outputs["Result"]
        else:
            producto = nt.nodes.new("ShaderNodeMath")
            producto.operation = 'MULTIPLY'
            producto.location = (-380, -160)
            nt.links.new(opacidad, producto.inputs[0])
            nt.links.new(fundido.outputs["Result"], producto.inputs[1])
            opacidad = producto.outputs[0]

    if opacidad is None:
        nt.links.new(emision.outputs["Emission"], salida.inputs["Surface"])
        return mat

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (-380, 140)
    mezcla = nt.nodes.new("ShaderNodeMixShader")
    mezcla.location = (-120, 0)
    nt.links.new(opacidad, mezcla.inputs[0])
    nt.links.new(transp.outputs[0], mezcla.inputs[1])
    nt.links.new(emision.outputs["Emission"], mezcla.inputs[2])
    nt.links.new(mezcla.outputs[0], salida.inputs["Surface"])
    return mat


def _al_cambiar_paisaje(self, contexto):
    ob = self.id_data
    if es_paisaje(ob):
        actualizar_paisaje(contexto.scene, ob)


def _al_cambiar_estilo_paisaje(self, contexto):
    ob = self.id_data
    if es_paisaje(ob):
        aplicar_estilo_paisaje(ob)


def _al_cambiar_color_paisaje(self, contexto):
    ob = self.id_data
    if es_paisaje(ob):
        actualizar_material_paisaje(ob)


# ---------------------------------------------------------------------------
# HORNEAR: DEJAR EL RESULTADO SIN DEPENDER DE LA EXTENSION
# ---------------------------------------------------------------------------
# Las barras, los LEDs y los cubos del compas NO hacen falta hornearlos: van con
# drivers y claves de animacion, que son de Blender. Comprobado abriendo un
# .blend sin la extension: siguen moviendose.
#
# El plexus y el paisaje si, porque su geometria la rehace Python en cada
# fotograma. Se hornean con CLAVES DE FORMA en modo absoluto: una por fotograma,
# y una sola curva ('eval_time') que las recorre. Todo son datablocks normales,
# asi que el resultado funciona en cualquier Blender.
#
# El paisaje sale identico, porque su rejilla no cambia de forma. El plexus
# congela las conexiones del fotograma de referencia; medido sobre el rango,
# con la amplitud normal el 98% de las aristas estan en todos los fotogramas y
# la coincidencia media es del 100%, asi que la diferencia no se ve. Con
# amplitudes muy grandes si cambia, y por eso el operador lo avisa.

SUFIJO_HORNEADO = "_Horneado"


def estabilidad_aristas(escena, ob, muestras=8):
    """Que porcentaje de las conexiones del fotograma actual siguen estando en
    el resto del rango. Sirve para avisar antes de hornear."""
    if len(ob.data.edges) == 0:
        return 100.0
    referencia = {tuple(sorted(e.vertices)) for e in ob.data.edges}
    guardado = escena.frame_current
    inicio, fin = escena.frame_start, escena.frame_end
    coincidencias = []
    try:
        for k in range(muestras):
            fr = int(inicio + (fin - inicio) * k / max(muestras - 1, 1))
            escena.frame_set(fr)
            actual = {tuple(sorted(e.vertices)) for e in ob.data.edges}
            if actual:
                coincidencias.append(len(referencia & actual) / len(actual) * 100.0)
    finally:
        escena.frame_set(guardado)
    return min(coincidencias) if coincidencias else 100.0


def hornear_objeto(escena, ob, inicio, fin, progreso=None):
    """Copia el objeto y le mete una clave de forma por fotograma.

    Devuelve el objeto horneado. El original se queda como estaba, para poder
    seguir tocandolo y volver a hornear.
    """
    guardado = escena.frame_current
    escena.frame_set(inicio)

    copia = ob.copy()
    copia.data = ob.data.copy()
    copia.name = ob.name + SUFIJO_HORNEADO
    copia.data.name = copia.name
    for coleccion in ob.users_collection:
        coleccion.objects.link(copia)

    # La copia no debe seguir siendo un plexus ni un paisaje: si no, el handler
    # la reconstruiria en cada fotograma y se cargaria lo horneado.
    if getattr(copia, "audioviz_plex", None) is not None:
        copia.audioviz_plex.es_plexus = False
        copia.audioviz_plex.objeto_caras = None
    if getattr(copia, "audioviz_paisaje", None) is not None:
        copia.audioviz_paisaje.es_paisaje = False
    if getattr(copia, "audioviz_enj", None) is not None:
        copia.audioviz_enj.es_enjambre = False
    for clave in (CLAVE_CACHE_P0, CLAVE_CACHE_DIR, CLAVE_CACHE_FIRMA,
                  CLAVE_FIRMA_PAISAJE, CLAVE_FIRMA_ENJAMBRE,
                  CLAVE_ENJ_RAD, CLAVE_ENJ_RADN, CLAVE_ENJ_ANG,
                  CLAVE_ENJ_ALT, CLAVE_ENJ_BANDA):
        if clave in copia:
            del copia[clave]

    n_verts = len(copia.data.vertices)
    copia.shape_key_add(name="Base", from_mix=False)
    plano = [0.0] * (n_verts * 3)
    total = fin - inicio + 1

    for indice, fr in enumerate(range(inicio, fin + 1)):
        escena.frame_set(fr)
        origen = ob.data
        if len(origen.vertices) != n_verts:
            # La densidad cambio a mitad de rango: no deberia pasar, pero mejor
            # avisar que escribir posiciones descuadradas.
            escena.frame_set(guardado)
            raise ValueError(f"en el fotograma {fr} el objeto tiene "
                             f"{len(origen.vertices)} vertices y no {n_verts}")
        origen.vertices.foreach_get("co", plano)
        clave = copia.shape_key_add(name=f"f{fr}", from_mix=False)
        clave.data.foreach_set("co", plano)
        clave.interpolation = 'KEY_LINEAR'
        if progreso is not None and indice % 16 == 0:
            progreso(indice, total)

    # Modo absoluto: las claves forman una secuencia y 'eval_time' la recorre,
    # asi que hay UNA sola curva animada en vez de una por fotograma.
    llaves = copia.data.shape_keys
    llaves.use_relative = False
    bloques = llaves.key_blocks

    # Una clave de animacion por fotograma, con su valor exacto. Con solo dos y
    # dejando que Blender interpole, el redondeo de la curva desviaba un pelin
    # el eval_time y el resultado no caia justo encima de cada forma: se colaba
    # una mezcla minuscula de la siguiente. Medido: 0.00024 de error. Asi es
    # exacto, y 300 claves en una curva no son nada.
    for indice, fr in enumerate(range(inicio, fin + 1)):
        llaves.eval_time = bloques[indice + 1].frame   # el bloque 0 es la base
        llaves.keyframe_insert("eval_time", frame=fr)

    ad = llaves.animation_data
    if ad is not None and ad.action is not None:
        for capa in ad.action.layers:
            for tira in capa.strips:
                bolsa = tira.channelbag(ad.action_slot)
                if bolsa is None:
                    continue
                for fc in bolsa.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
                    fc.update()

    escena.frame_set(guardado)
    return copia


def es_plexus(ob):
    return (ob is not None and ob.type == 'MESH'
            and getattr(ob, "audioviz_plex", None) is not None
            and ob.audioviz_plex.es_plexus)


def plexus_de_la_escena(escena):
    return [o for o in escena.objects if es_plexus(o)]


def valores_de_bandas(empty, fotograma, n_bandas, canal='MONO'):
    """Valor de cada banda en un fotograma, leido de las curvas de animacion.

    Leemos la curva en vez de la propiedad del objeto a proposito: cuando el
    handler se dispara, Blender aun no ha evaluado la animacion del fotograma
    nuevo, asi que la propiedad todavia tendria el valor del fotograma anterior
    y el plexus iria un fotograma por detras. La curva se puede consultar en
    cualquier momento y da el valor exacto.
    """
    canal = canal_util(empty, canal)
    prefijo = PREFIJOS_CANAL.get(canal, PREFIJO_BANDA)
    curvas = curvas_de_bandas(empty, n_bandas, canal)
    if curvas is None:
        return [float(empty.get(f"{prefijo}{i}", 0.0)) for i in range(n_bandas)]
    return [float(fc.evaluate(fotograma)) for fc in curvas]


def mezcla_estereo(p0):
    """Peso del canal derecho segun la posicion en X: 0 a la izquierda, 1 a la
    derecha, con una transicion suave por el centro para que no haya costura."""
    x = p0[:, 0]
    ancho = float(x.max() - x.min())
    if ancho < 1e-9:
        return np.full(len(p0), 0.5)
    t = (x - x.min()) / ancho
    # Suavizado de Hermite: plano en los extremos, todo el cambio en el centro.
    return t * t * (3.0 - 2.0 * t)


def valores_por_canal(fuente, fotograma, n_bandas, canal):
    """Valores de banda en ese fotograma. En estereo devuelve los dos juegos."""
    canal = canal_util(fuente, canal)
    if canal != 'ESTEREO':
        return valores_de_bandas(fuente, fotograma, n_bandas, canal), None
    return (valores_de_bandas(fuente, fotograma, n_bandas, 'IZQ'),
            valores_de_bandas(fuente, fotograma, n_bandas, 'DER'))


def _distancia_normalizada(p0):
    """0 en el centro, 1 en el borde. Por donde va pasando la onda.

    En una esfera o un anillo todos los puntos estan a la misma distancia del
    centro, asi que ahi no sirve: se pasa a la altura, y si tampoco varia, al
    orden de los puntos (que en un anillo equivale a dar la vuelta).
    """
    for metrica in (np.linalg.norm(p0, axis=1), p0[:, 2],
                    np.arange(len(p0), dtype=np.float64)):
        bajo, alto = float(metrica.min()), float(metrica.max())
        if alto - bajo > 1e-6:
            return (metrica - bajo) / (alto - bajo)
    return np.zeros(len(p0))


def reconstruir_plexus(escena, ob):
    """Recalcula posiciones y conexiones de UN plexus. Devuelve n aristas."""
    if not es_plexus(ob) or np is None:
        return -1

    p = ob.audioviz_plex
    # Cada plexus escucha la fuente de audio que tenga asignada. El poll del
    # desplegable no basta: hay que comprobarlo aqui.
    empty = p.fuente if es_fuente(p.fuente) else fuente_activa(escena)
    if empty is None:
        return -1

    n_bandas = int(empty.get(CLAVE_BANDAS, 8))

    p0, dirs = disposicion(ob)
    n = len(p0)
    bandas = asignar_bandas(p0, p.asignacion, n_bandas, p.banda_min, p.banda_max, p.semilla)

    v_uno, v_dos = valores_por_canal(empty, escena.frame_current, n_bandas, p.canal)

    def a_puntos(valores):
        valores = np.asarray(valores, dtype=float)
        if p.suave:
            # Un punto en la posicion 3.7 del espectro recibe el 30% de la banda
            # 3 y el 70% de la banda 4: la superficie queda continua.
            return np.interp(bandas, np.arange(n_bandas), valores)
        # Cada punto se queda con la banda mas cercana: mesetas y escalones,
        # que es el aspecto de ecualizador de toda la vida.
        return valores[np.clip(np.round(bandas).astype(int), 0, n_bandas - 1)]

    crudo = a_puntos(v_uno)
    if v_dos is not None:
        # Estereo: la mitad izquierda de la nube sigue al canal izquierdo, la
        # derecha al derecho, y por el centro se funden.
        peso = mezcla_estereo(p0)
        crudo = crudo * (1.0 - peso) + a_puntos(v_dos) * peso

    desplazamiento = crudo * p.amplitud
    if tiene_compas(empty):
        if p.pulso_amplitud != 0.0:
            # Un empujon de toda la nube a la vez en cada golpe.
            fc = curva_por_ruta(empty, f'["{CLAVE_PULSO}"]')
            if fc is not None:
                desplazamiento = desplazamiento + \
                    fc.evaluate(escena.frame_current) * p.pulso_amplitud

        # Ondas expansivas: anillos que salen del centro y llegan al borde justo
        # al acabar su ciclo. Se leen como ritmo mucho mejor que un latido
        # uniforme, que solo parece que el objeto engorda.
        #
        # Hay dos porque van a ritmos distintos y se pueden superponer: la del
        # pulso cruza en cada golpe, la del compas tarda un compas entero. Juntas
        # dan dos capas, una rapida y una de fondo.
        if p.pulso_onda != 0.0 or p.compas_onda != 0.0:
            d = _distancia_normalizada(p0)
            for clave, amplitud, grosor in (
                    (CLAVE_FASE_PULSO, p.pulso_onda, p.onda_grosor),
                    (CLAVE_FASE_COMPAS, p.compas_onda, p.compas_onda_grosor)):
                if amplitud == 0.0:
                    continue
                fc = curva_por_ruta(empty, f'["{clave}"]')
                if fc is None:
                    continue
                fase = float(fc.evaluate(escena.frame_current))    # 0..1 del ciclo
                anillo = np.exp(-((d - fase) / max(grosor, 1e-3)) ** 2)
                desplazamiento = desplazamiento + anillo * amplitud

    puntos = p0 + dirs * desplazamiento[:, None]

    # --- que puntos se unen con que ---
    # Distancias al cuadrado por el truco algebraico |a-b|^2 = |a|^2 + |b|^2 - 2ab:
    # una multiplicacion de matrices en vez de un bucle de n*n en Python.
    cuad = (puntos ** 2).sum(axis=1)
    d2 = cuad[:, None] + cuad[None, :] - 2.0 * (puntos @ puntos.T)
    np.fill_diagonal(d2, np.inf)   # un punto no se une consigo mismo

    k = max(1, min(int(p.conexiones), n - 1))
    # argpartition saca los k mas cercanos sin ordenarlo todo: mucho mas rapido.
    vecinos = np.argpartition(d2, k - 1, axis=1)[:, :k]

    filas = np.repeat(np.arange(n), k)
    columnas = vecinos.ravel()
    dentro = d2[filas, columnas] <= p.distancia ** 2
    pares = np.stack([filas[dentro], columnas[dentro]], axis=1)

    if len(pares):
        pares = np.unique(np.sort(pares, axis=1), axis=0)   # (a,b) y (b,a) son la misma

    # --- reescribir la malla ---
    me = ob.data
    me.clear_geometry()
    me.vertices.add(n)
    me.vertices.foreach_set("co", puntos.astype(np.float32).ravel())
    if len(pares):
        me.edges.add(len(pares))
        me.edges.foreach_set("vertices", pares.astype(np.int32).ravel())

    # --- atributos que puede leer el material ---
    # av_nivel      : DONDE esta el punto en el espectro (0 grave, 1 agudo). Fijo.
    # av_intensidad : CUANTO suena su banda en este fotograma (0..1). Cambia.
    niveles = np.clip(bandas / max(n_bandas - 1, 1), 0.0, 1.0).astype(np.float32).ravel()
    intensidades = np.clip(crudo, 0.0, 1.0).astype(np.float32).ravel()
    poner_atributo(me, "av_nivel", niveles)
    poner_atributo(me, "av_intensidad", intensidades)

    me.update()

    # --- caras, en su objeto aparte ---
    if p.caras:
        escribir_caras(ob, puntos, pares, niveles, intensidades)
    return len(pares)


# ---------------------------------------------------------------------------
# PRESET 4: ENJAMBRE ORBITAL
# ---------------------------------------------------------------------------
# Una nube de miles de particulas girando alrededor de un centro. Cada una
# pertenece a una banda: cuando esa frecuencia suena, sus particulas se van
# hacia fuera y se encienden. En cada pulso sale una onda expansiva que las
# barre, igual que en el plexus.
#
# LO IMPORTANTE DEL DISENO: aqui no hay simulacion. La posicion de una particula
# en el fotograma 4000 se calcula con una formula, sin haber pasado por el 3999.
# Eso es lo que permite arrastrar la barra de tiempo a donde quieras y que sea
# correcto, hornear con el boton de siempre, y renderizar en segundo plano sin
# cache que se pueda corromper. Un sistema de particulas de verdad daria mas
# variedad de movimiento, pero perderia las tres cosas.
#
# La contrapartida: no hay choques, ni turbulencia acumulada, ni particulas que
# recuerden por donde han pasado. El movimiento sale de combinar un giro, un
# empuje radial y las ondas, que para leer musica da de sobra.

FORMAS_ENJAMBRE = (
    ('DISCO', "Disco", "Plano, tipo galaxia. Es el que mejor lee el espectro de un vistazo"),
    ('ESFERA', "Esfera", "Nube en volumen. Envuelve mas, pero tapa sus propias particulas"),
    ('ANILLO', "Anillo", "Solo el borde, con el centro vacio"),
)

REPARTOS_ENJAMBRE = (
    ('RADIO', "Del centro al borde",
     "Los graves en el centro y los agudos fuera. El espectro se lee como una diana"),
    ('ANGULO', "Alrededor",
     "El espectro da la vuelta al circulo, como las horas de un reloj"),
    ('AZAR', "Mezcladas",
     "Cada particula coge una banda al azar. Parece un enjambre de verdad, "
     "pero se pierde el ecualizador"),
)


def es_enjambre(ob):
    return (ob is not None and ob.type == 'MESH'
            and getattr(ob, "audioviz_enj", None) is not None
            and ob.audioviz_enj.es_enjambre)


def enjambres_de_la_escena(escena):
    return [o for o in escena.objects if es_enjambre(o)]


def firma_enjambre(p):
    return (f"{p.particulas}|{p.forma}|{p.semilla}|{p.grosor:.4f}|{p.reparto}|"
            f"{p.banda_min}|{p.banda_max}")


def disposicion_enjambre(ob, forzar=False):
    """Las cuatro tablas fijas de la nube, cacheadas en el objeto.

    Se generan una vez y no cambian con el fotograma: lo que cambia cada
    fotograma es lo que se hace con ellas. Devuelve, por particula:
      rad   distancia al centro, en unidades de radio (0..1 aprox)
      radn  la misma distancia normalizada a 0..1 exactos, para las ondas
      ang   angulo de partida alrededor del eje
      alt   altura (o coseno de la inclinacion, en la esfera)
      banda posicion dentro del espectro, en continuo (p.ej. 3.7)
    """
    p = ob.audioviz_enj
    firma = firma_enjambre(p)

    if not forzar and ob.get(CLAVE_FIRMA_ENJAMBRE) == firma:
        try:
            tablas = [np.array(ob[c], dtype=np.float64) for c in
                      (CLAVE_ENJ_RAD, CLAVE_ENJ_RADN, CLAVE_ENJ_ANG,
                       CLAVE_ENJ_ALT, CLAVE_ENJ_BANDA)]
            if len(tablas[0]) == max(int(p.particulas), 8):
                return tablas
        except Exception:
            pass    # cache corrupto: lo rehacemos

    n = max(int(p.particulas), 8)
    rng = np.random.default_rng(int(p.semilla))
    u = rng.random(n)
    ang = rng.random(n) * 2.0 * np.pi

    # Campana recortada a dos desviaciones: la de Gauss a secas no tiene tope, y
    # con unos miles de particulas siempre sale alguna disparada lejisimos. Con
    # esto el grosor es una promesa: 'grosor' es la mitad del espesor en
    # fraccion del radio, y ninguna se sale de ahi.
    def campana():
        return np.clip(rng.standard_normal(n), -2.0, 2.0) * 0.5 * p.grosor

    if p.forma == 'ESFERA':
        # Raiz cubica para que el volumen quede parejo: sin ella se amontonarian
        # todas en el centro, que es donde caben menos.
        rad = np.cbrt(u)
        alt = rng.random(n) * 2.0 - 1.0        # coseno de la inclinacion
    elif p.forma == 'ANILLO':
        rad = 1.0 + campana()
        alt = campana()
    else:   # DISCO: raiz cuadrada, por lo mismo pero en area
        rad = np.sqrt(u)
        alt = campana()

    bajo, alto = float(rad.min()), float(rad.max())
    radn = (rad - bajo) / (alto - bajo) if alto - bajo > 1e-9 else np.zeros(n)

    if p.reparto == 'ANGULO':
        t_banda = ang / (2.0 * np.pi)
    elif p.reparto == 'AZAR':
        t_banda = rng.random(n)
    else:   # RADIO
        t_banda = radn
    b0, b1 = int(p.banda_min), int(p.banda_max)
    if b1 < b0:
        b0, b1 = b1, b0
    banda = b0 + t_banda * (b1 - b0)

    tablas = [rad, radn, ang, alt, banda]
    for clave, tabla in zip((CLAVE_ENJ_RAD, CLAVE_ENJ_RADN, CLAVE_ENJ_ANG,
                             CLAVE_ENJ_ALT, CLAVE_ENJ_BANDA), tablas):
        ob[clave] = tabla.tolist()
    ob[CLAVE_FIRMA_ENJAMBRE] = firma
    return tablas


def nucleo_inercia(fps, vuelta, rebote):
    """La 'huella' que deja un golpe seco en una particula con inercia.

    Es la respuesta de una masa colgada de un muelle con amortiguador: le pegas
    un empujon y no salta y vuelve de golpe, sino que sale, frena y regresa,
    rebotando mas o menos segun lo blando que sea el amortiguador.

    Devolver esa huella como una tabla es el truco que permite tener inercia sin
    simular: en vez de arrastrar la velocidad de un fotograma al siguiente, se
    mira hacia atras y se suma lo que ha ido empujando cada golpe reciente. El
    resultado es identico y no depende de por donde hayas llegado al fotograma.
    """
    # rebote 0 -> vuelve sin pasarse; 1 -> se pasa y oscila varias veces
    zeta = float(np.clip(1.0 - rebote * 0.85, 0.15, 0.99))
    w = 2.0 * np.pi / max(float(vuelta), 1e-3)
    wd = w * np.sqrt(max(1.0 - zeta ** 2, 1e-6))

    # Miramos atras hasta que el golpe ya no se nota (cuatro constantes de
    # tiempo deja menos del 2%). Con topes, que un tema puede durar 10 minutos.
    largo = int(np.clip(4.0 / (zeta * w) * fps, 4, 400))
    t = np.arange(largo) / max(fps, 1e-6)
    h = np.exp(-zeta * w * t) * np.sin(wd * t)
    total = h.sum()
    # Normalizado para que la fuerza signifique algo concreto: con una banda
    # sonando a tope y sostenida, la particula acaba desplazandose justo lo que
    # diga el mando, ni mas ni menos.
    return h / total if abs(total) > 1e-9 else h


def fuerzas_inerciales(empty, escena, n_bandas, canal, vuelta, rebote, fps):
    """Cuanto ha desplazado a cada BANDA lo que ha sonado ultimamente."""
    h = nucleo_inercia(fps, vuelta, rebote)
    fr = escena.frame_current
    canal = canal_util(empty, canal)

    def responde(c):
        curvas = curvas_de_bandas(empty, n_bandas, c)
        tabla = np.array([[cu.evaluate(fr - k) if cu is not None else 0.0
                           for k in range(len(h))] for cu in curvas])
        return tabla @ h

    if canal != 'ESTEREO':
        return responde(canal), None
    return responde('IZQ'), responde('DER')


def pesos_espectro(banda, n_bandas, graves, medios, agudos):
    """Cuanto manda cada particula segun en que zona del espectro caiga.

    Tres ventanas triangulares que se solapan y SUMAN UNO en todo el recorrido.
    Esa propiedad es la que hace que los mandos se entiendan solos: con los tres
    a 1 no pasa nada, y con los tres a 2 todo el mundo empuja el doble. Y como en
    el extremo grave solo pesa 'graves', en el centro solo 'medios' y en el
    agudo solo 'agudos', cada mando hace exactamente lo que dice su nombre.

    Tambien sustituye a dos mandos del paisaje: bajar graves y subir agudos es
    la inclinacion, y subir los tres a la vez es la ganancia.
    """
    e = np.clip(banda / max(n_bandas - 1, 1), 0.0, 1.0)
    w_gra = np.clip(1.0 - 2.0 * e, 0.0, 1.0)
    w_agu = np.clip(2.0 * e - 1.0, 0.0, 1.0)
    w_med = 1.0 - w_gra - w_agu
    return w_gra * graves + w_med * medios + w_agu * agudos


def turbulencia_enjambre(p0, t, escala, velocidad, semilla):
    """Un campo de remolinos suave por el que las particulas van nadando.

    Tres capas de ondas de tamano decreciente. Es ruido barato, pero se mueve
    despacio y a la vez en zonas enteras, que es lo que lo hace parecer una
    corriente y no un temblor: dos particulas vecinas reciben casi el mismo
    empujon, asi que se mueven juntas.
    """
    rng = np.random.default_rng(int(semilla) + 7919)
    salida = np.zeros_like(p0)
    amplitud, frecuencia, suma = 1.0, 1.0 / max(escala, 1e-3), 0.0
    for _ in range(3):
        # Una direccion y una deriva propias para cada eje de salida: si
        # compartieran las mismas, los remolinos saldrian en diagonal perfecta.
        dirs = rng.standard_normal((3, 3))
        fases = rng.random(3) * 2.0 * np.pi
        deriva = rng.standard_normal(3)
        for eje in range(3):
            salida[:, eje] += amplitud * np.sin(
                p0 @ dirs[eje] * frecuencia + deriva[eje] * velocidad * t + fases[eje])
        suma += amplitud
        amplitud *= 0.5
        frecuencia *= 2.1
    return salida / max(suma, 1e-9)


def posiciones_enjambre(p, rad, ang, alt, radio):
    """De coordenadas polares a XYZ, segun la forma."""
    if p.forma == 'ESFERA':
        cos_i = np.clip(alt, -1.0, 1.0)
        sen_i = np.sqrt(np.maximum(1.0 - cos_i ** 2, 0.0))
        return np.stack([radio * rad * sen_i * np.cos(ang),
                         radio * rad * sen_i * np.sin(ang),
                         radio * rad * cos_i], axis=1)
    return np.stack([radio * rad * np.cos(ang),
                     radio * rad * np.sin(ang),
                     radio * alt], axis=1)


def reconstruir_enjambre(escena, ob):
    """Recoloca las particulas de UN enjambre. Devuelve cuantas son."""
    if not es_enjambre(ob) or np is None:
        return -1

    p = ob.audioviz_enj
    empty = p.fuente if es_fuente(p.fuente) else fuente_activa(escena)
    if empty is None:
        return -1

    n_bandas = int(empty.get(CLAVE_BANDAS, 8))
    rad, radn, ang0, alt, banda = disposicion_enjambre(ob)
    n = len(rad)

    # --- giro ---
    # El tiempo se cuenta en segundos desde el principio de la escena, no en
    # fotogramas: asi el giro es el mismo tanto si trabajas a 24 como a 60 fps.
    fps = float(empty.audioviz_audio.fps) or float(escena.render.fps)
    t = (escena.frame_current - escena.frame_start) / max(fps, 1e-6)
    # Las de dentro giran mas rapido que las de fuera, como en una galaxia. Con
    # todas a la misma velocidad la nube parece un solido girando.
    velocidad = p.giro * 2.0 * np.pi * (1.0 + p.diferencial * (1.0 - radn))
    ang = ang0 + velocidad * t

    # --- cuanto suena la banda de cada particula ---
    p0 = posiciones_enjambre(p, rad, ang, alt, p.radio)
    v_uno, v_dos = valores_por_canal(empty, escena.frame_current, n_bandas, p.canal)

    def a_particulas(valores):
        valores = np.asarray(valores, dtype=float)
        if p.suave:
            return np.interp(banda, np.arange(n_bandas), valores)
        return valores[np.clip(np.round(banda).astype(int), 0, n_bandas - 1)]

    intens = a_particulas(v_uno)
    if v_dos is not None:
        peso = mezcla_estereo(p0)
        intens = intens * (1.0 - peso) + a_particulas(v_dos) * peso

    # Cuanto manda cada zona del espectro. Se aplica aqui, antes de repartir a
    # los empujes y al material, para que un grave al que le has bajado el peso
    # ni mueva ni encienda: si no, la nube diria una cosa y se veria otra.
    pesos_banda = pesos_espectro(banda, n_bandas, p.peso_graves, p.peso_medios,
                                 p.peso_agudos)
    intens = intens * pesos_banda

    # --- las frecuencias como FUERZA, no como posicion ---
    # El empuje de arriba coloca la particula donde diga el audio ahora mismo:
    # el audio salta, la particula salta. Esto otro trata cada golpe como un
    # empujon sobre algo que pesa, asi que sale despacio y vuelve sola. Los dos
    # se pueden mezclar; con el empuje a 0 y esto subido, el movimiento es mucho
    # mas sutil y no da la sensacion de temblar.
    inercia = 0.0
    if p.fuerza != 0.0:
        f_uno, f_dos = fuerzas_inerciales(empty, escena, n_bandas, p.canal,
                                          p.vuelta, p.rebote, fps)
        inercia = a_particulas(f_uno)
        if f_dos is not None:
            inercia = inercia * (1.0 - peso) + a_particulas(f_dos) * peso
        # Los mismos pesos. Se pueden aplicar despues de mirar hacia atras
        # porque la respuesta es lineal: pesar y luego sumar da igual que
        # sumar y luego pesar, y asi se hace una vez en vez de por fotograma.
        inercia = inercia * pesos_banda

    # --- empuje y ondas ---
    radio_actual = p.radio * rad + p.empuje * intens + p.fuerza * inercia
    golpe = np.zeros(n)
    if tiene_compas(empty):
        for clave, amplitud, grosor in (
                (CLAVE_FASE_PULSO, p.pulso_onda, p.onda_grosor),
                (CLAVE_FASE_COMPAS, p.compas_onda, p.compas_onda_grosor)):
            if amplitud == 0.0:
                continue
            fc = curva_por_ruta(empty, f'["{clave}"]')
            if fc is None:
                continue
            fase = float(fc.evaluate(escena.frame_current))
            anillo = np.exp(-((radn - fase) / max(grosor, 1e-3)) ** 2)
            radio_actual = radio_actual + anillo * amplitud
            # El maximo, no la suma: si las dos ondas coinciden no queremos un
            # destello del doble, queremos que se note que han coincidido.
            golpe = np.maximum(golpe, anillo)

    puntos = posiciones_enjambre(p, radio_actual / max(p.radio, 1e-6), ang, alt, p.radio)

    if p.turbulencia != 0.0:
        # Lo que reacciona al audio es CUANTO agita, no lo rapido que se mueve el
        # campo. Acelerarlo con la musica parece la idea obvia y es justo la
        # trampa: al cambiar la velocidad cambia la fase de golpe y las
        # particulas dan un salto en cada golpe. Subiendo la amplitud, la
        # corriente sigue siendo la misma y solo aprieta mas.
        agita = np.full(n, float(p.turbulencia))
        if p.turb_audio > 0.0:
            agita = agita * ((1.0 - p.turb_audio)
                             + p.turb_audio * np.clip(intens, 0.0, 1.0))
        # El campo se muestrea en la posicion de REPOSO, no en la desplazada: si
        # no, una particula empujada lejos cruzaria el ruido a toda velocidad y
        # volveria el temblor que precisamente queriamos quitar.
        puntos = puntos + turbulencia_enjambre(
            p0, t, p.turb_escala, p.turb_velocidad, p.semilla) * agita[:, None]

    # --- a la malla ---
    me = ob.data
    me.clear_geometry()
    me.vertices.add(n)
    me.vertices.foreach_set("co", puntos.astype(np.float32).ravel())

    # Los mismos nombres que en los otros presets, para que un material que
    # hayas montado para el plexus valga aqui tal cual.
    poner_atributo(me, "av_nivel",
                   np.clip(banda / max(n_bandas - 1, 1), 0.0, 1.0)
                   .astype(np.float32).ravel())
    poner_atributo(me, "av_intensidad", np.clip(intens, 0.0, 1.0)
                   .astype(np.float32).ravel())
    # Este es propio del enjambre: cuanto le esta dando la onda AHORA. Sirve
    # para que la onda no solo empuje, sino que ademas encienda por donde pasa.
    poner_atributo(me, "av_golpe", np.clip(golpe, 0.0, 1.0)
                   .astype(np.float32).ravel())
    me.update()
    return n


def obtener_nodos_enjambre():
    """Grupo de nodos compartido: pone una bolita en cada particula.

    El tamano de cada bolita sale del atributo av_intensidad, asi que las
    particulas de la banda que esta sonando crecen ademas de encenderse.
    """
    ng = bpy.data.node_groups.get(NOMBRE_GN_ENJAMBRE)
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(NOMBRE_GN_ENJAMBRE, "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Tamano", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket(name="Reaccion", in_out='INPUT', socket_type='NodeSocketFloat')
    ng.interface.new_socket(name="Material", in_out='INPUT', socket_type='NodeSocketMaterial')

    nodos, enlaces = ng.nodes, ng.links
    entrada = nodos.new("NodeGroupInput"); entrada.location = (-760, 0)
    salida = nodos.new("NodeGroupOutput"); salida.location = (400, 0)

    bola = nodos.new("GeometryNodeMeshIcoSphere"); bola.location = (-540, -210)
    bola.inputs["Subdivisions"].default_value = 1     # 42 vertices: de sobra
    bola.inputs["Radius"].default_value = 1.0
    # Sin esto las bolitas se leen como hexagonos cuando se ven grandes. Suavizar
    # el sombreado sale gratis; subir las subdivisiones costaria cuatro veces mas
    # geometria por cada una de las miles de particulas.
    suave = nodos.new("GeometryNodeSetShadeSmooth"); suave.location = (-400, -210)
    mat = nodos.new("GeometryNodeSetMaterial"); mat.location = (-250, -210)
    enlaces.new(bola.outputs["Mesh"], suave.inputs["Geometry"])
    enlaces.new(suave.outputs["Geometry"], mat.inputs["Geometry"])
    enlaces.new(entrada.outputs["Material"], mat.inputs["Material"])

    # tamano = Tamano * (1 + Reaccion * intensidad)
    leer = nodos.new("GeometryNodeInputNamedAttribute"); leer.location = (-760, 240)
    leer.data_type = 'FLOAT'
    leer.inputs["Name"].default_value = "av_intensidad"
    m1 = nodos.new("ShaderNodeMath"); m1.location = (-540, 240)
    m1.operation = 'MULTIPLY'
    m2 = nodos.new("ShaderNodeMath"); m2.location = (-330, 240)
    m2.operation = 'ADD'
    m2.inputs[1].default_value = 1.0
    m3 = nodos.new("ShaderNodeMath"); m3.location = (-120, 240)
    m3.operation = 'MULTIPLY'
    enlaces.new(leer.outputs["Attribute"], m1.inputs[0])
    enlaces.new(entrada.outputs["Reaccion"], m1.inputs[1])
    enlaces.new(m1.outputs[0], m2.inputs[0])
    enlaces.new(m2.outputs[0], m3.inputs[0])
    enlaces.new(entrada.outputs["Tamano"], m3.inputs[1])

    instanciar = nodos.new("GeometryNodeInstanceOnPoints"); instanciar.location = (120, 0)
    enlaces.new(entrada.outputs[0], instanciar.inputs["Points"])
    enlaces.new(mat.outputs["Geometry"], instanciar.inputs["Instance"])
    enlaces.new(m3.outputs[0], instanciar.inputs["Scale"])
    enlaces.new(instanciar.outputs["Instances"], salida.inputs[0])
    return ng


def modificador_enjambre(ob):
    for m in ob.modifiers:
        if m.type == 'NODES' and m.node_group is not None \
                and m.node_group.name == NOMBRE_GN_ENJAMBRE:
            return m
    return None


def aplicar_estilo_enjambre(ob):
    m = modificador_enjambre(ob)
    if m is None:
        return
    p = ob.audioviz_enj
    for nombre, valor in (("Tamano", p.tam_punto), ("Reaccion", p.reaccion_tam)):
        ident = identificador_entrada(m.node_group, nombre)
        if ident is not None:
            m[ident] = valor
    ob.update_tag()


def material_de_enjambre(ob, crear=False):
    m = modificador_enjambre(ob)
    if m is None:
        return None
    ident = identificador_entrada(m.node_group, "Material")
    if ident is None:
        return None
    mat = m[ident]
    if mat is None and crear:
        mat = bpy.data.materials.new(NOMBRE_MAT_ENJAMBRE)
        m[ident] = mat
    return mat


def actualizar_material_enjambre(ob):
    """Shader del enjambre: color por banda, brillo por lo que suena.

    Los atributos se leen en modo INSTANCER y no GEOMETRY: la bolita que se ve
    es una copia instanciada, y los valores viven en el punto que la instancia,
    no en la bolita.
    """
    mat = material_de_enjambre(ob, crear=True)
    if mat is None:
        return
    if mat.node_tree is None:
        mat.use_nodes = True
    p = ob.audioviz_enj

    nt = mat.node_tree
    nt.nodes.clear()

    nivel = nt.nodes.new("ShaderNodeAttribute")
    nivel.attribute_type = 'INSTANCER'
    nivel.attribute_name = "av_nivel"
    nivel.location = (-820, 120)

    rampa = nt.nodes.new("ShaderNodeValToRGB"); rampa.location = (-620, 120)
    rampa.color_ramp.elements[0].position = 0.0
    rampa.color_ramp.elements[0].color = (*p.color_grave, 1.0)
    rampa.color_ramp.elements[1].position = 1.0
    rampa.color_ramp.elements[1].color = (*p.color_agudo, 1.0)
    rampa.color_ramp.elements.new(0.5).color = (*p.color_medio, 1.0)

    intens = nt.nodes.new("ShaderNodeAttribute")
    intens.attribute_type = 'INSTANCER'
    intens.attribute_name = "av_intensidad"
    intens.location = (-820, -120)

    golpe = nt.nodes.new("ShaderNodeAttribute")
    golpe.attribute_type = 'INSTANCER'
    golpe.attribute_name = "av_golpe"
    golpe.location = (-820, -300)

    # brillo = ((intensidad + golpe*destello) + fondo) * brillo
    m1 = nt.nodes.new("ShaderNodeMath"); m1.location = (-620, -300)
    m1.operation = 'MULTIPLY'
    m1.inputs[1].default_value = p.destello
    m2 = nt.nodes.new("ShaderNodeMath"); m2.location = (-430, -180)
    m2.operation = 'ADD'
    m3 = nt.nodes.new("ShaderNodeMath"); m3.location = (-240, -180)
    m3.operation = 'ADD'
    # Un suelo pequeno para que las particulas calladas no desaparezcan del todo
    # y se siga viendo la forma de la nube.
    m3.inputs[1].default_value = p.fondo
    m4 = nt.nodes.new("ShaderNodeMath"); m4.location = (-50, -180)
    m4.operation = 'MULTIPLY'
    m4.inputs[1].default_value = p.brillo

    emision = nt.nodes.new("ShaderNodeEmission"); emision.location = (160, 0)
    emision.name = "Emision"
    salida = nt.nodes.new("ShaderNodeOutputMaterial"); salida.location = (360, 0)

    nt.links.new(nivel.outputs["Factor"], rampa.inputs["Fac"])
    nt.links.new(rampa.outputs["Color"], emision.inputs["Color"])
    nt.links.new(golpe.outputs["Factor"], m1.inputs[0])
    nt.links.new(intens.outputs["Factor"], m2.inputs[0])
    nt.links.new(m1.outputs[0], m2.inputs[1])
    nt.links.new(m2.outputs[0], m3.inputs[0])
    nt.links.new(m3.outputs[0], m4.inputs[0])
    nt.links.new(m4.outputs[0], emision.inputs["Strength"])
    nt.links.new(emision.outputs["Emission"], salida.inputs["Surface"])
    return mat


# Mientras se renderiza no sirve mirar si algo se ve en el viewport: un objeto
# puede estar oculto ahi y salir en el render igualmente. Estos dos handlers
# llevan la cuenta de en que estamos.
_renderizando = False


@persistent
def _empieza_render(escena, dg=None):
    global _renderizando
    _renderizando = True


@persistent
def _acaba_render(escena, dg=None):
    global _renderizando
    _renderizando = False


def hace_falta_actualizar(ob):
    """Si no se va a ver, no hay por que recalcularlo."""
    if _renderizando:
        return not ob.hide_render
    try:
        # visible_get() tiene en cuenta el ojo, el monitor y las colecciones.
        return ob.visible_get()
    except Exception:
        return not ob.hide_viewport


@persistent
def _al_cambiar_fotograma(escena, dg=None):
    """Se dispara sola en cada cambio de fotograma, tambien durante el render.

    Recorre TODOS los plexus y paisajes de la escena: cada uno se reconstruye
    con su propia configuracion, que vive en el objeto y no en la escena. Los
    que no se ven se saltan; si tienes tres montados y solo enseñas uno,
    pagabas los tres en cada fotograma.
    """
    for ob in plexus_de_la_escena(escena):
        if not hace_falta_actualizar(ob):
            continue
        try:
            reconstruir_plexus(escena, ob)
        except Exception as e:
            print(f"Audio Viz: fallo al reconstruir {ob.name}: {e}")

    for ob in paisajes_de_la_escena(escena):
        if not hace_falta_actualizar(ob):
            continue
        try:
            actualizar_paisaje(escena, ob)
        except Exception as e:
            print(f"Audio Viz: fallo al actualizar {ob.name}: {e}")

    for ob in enjambres_de_la_escena(escena):
        if not hace_falta_actualizar(ob):
            continue
        try:
            reconstruir_enjambre(escena, ob)
        except Exception as e:
            print(f"Audio Viz: fallo al recolocar {ob.name}: {e}")

    # La linea del fotograma en el espectrograma. Solo repinta, no recalcula.
    if _espectros:
        try:
            actualizar_marca_espectro(escena)
        except Exception as e:
            print(f"Audio Viz: fallo al mover la marca del analisis: {e}")


# En los tres callbacks siguientes, `self` es la configuracion pegada a UN
# objeto, y self.id_data es ese objeto. Asi cada deslizador afecta solo al
# plexus que estas editando y no a los demas.

def _al_cambiar_plexus(self, contexto):
    ob = self.id_data
    if es_plexus(ob):
        reconstruir_plexus(contexto.scene, ob)


def _al_cambiar_estilo_plexus(self, contexto):
    ob = self.id_data
    if es_plexus(ob):
        aplicar_estilo_plexus(ob)


def _al_cambiar_color_plexus(self, contexto):
    ob = self.id_data
    if es_plexus(ob):
        actualizar_material_plexus(ob)


def _al_cambiar_caras(self, contexto):
    ob = self.id_data
    if not es_plexus(ob):
        return
    if self.caras:
        objeto_caras_de(ob, crear=True)
        reconstruir_plexus(contexto.scene, ob)
    else:
        quitar_objeto_caras(ob)


def _al_cambiar_color_caras(self, contexto):
    ob = self.id_data
    if es_plexus(ob):
        actualizar_material_caras(ob)


def _al_cambiar_origen(self, contexto):
    """Al elegir un modelo, adaptamos amplitud y distancia a su tamano.

    Un modelo de la escena puede medir 0.1 o 100 unidades. Con los valores que
    hubiera puestos antes lo normal seria ver una mancha solida, polvo suelto o
    directamente una explosion: si la amplitud del audio es mayor que el propio
    modelo, el desplazamiento manda los puntos mucho mas lejos de lo que miden
    las uniones y no queda ni una linea.
    """
    ob = self.id_data
    if not es_plexus(ob):
        return
    reconstruir_plexus(contexto.scene, ob)
    if self.objeto_origen is None or self.forma not in FORMAS_DE_MODELO:
        return

    diagonal = tamano_disposicion(ob)
    if diagonal > 0.0:
        # El 7% de la diagonal del modelo. Comprobado renderizando: por encima
        # del 10% la silueta se desintegra y ya no se reconoce que modelo es.
        self.amplitud = diagonal * 0.07
    sugerida = distancia_sugerida(ob)
    if sugerida > 0.0:
        self.distancia = sugerida   # su propio update rehace el plexus


def _al_cambiar_enjambre(self, contexto):
    ob = self.id_data
    if es_enjambre(ob):
        reconstruir_enjambre(contexto.scene, ob)


def _al_cambiar_estilo_enjambre(self, contexto):
    ob = self.id_data
    if es_enjambre(ob):
        aplicar_estilo_enjambre(ob)


def _al_cambiar_color_enjambre(self, contexto):
    ob = self.id_data
    if es_enjambre(ob):
        actualizar_material_enjambre(ob)


def _poll_modelo(self, ob):
    # Cualquier malla menos las que genera el propio plugin: muestrear un plexus
    # a partir de si mismo no lleva a ningun sitio bueno.
    return (ob.type == 'MESH' and not es_plexus(ob) and not es_paisaje(ob)
            and not ob.name.endswith(SUFIJO_CARAS))


# ---------------------------------------------------------------------------
# PROPIEDADES DEL PANEL
# ---------------------------------------------------------------------------

class AV_AudioAjustes(PropertyGroup):
    """Ajustes de UNA fuente de audio. Se registra en bpy.types.Object, asi que
    cada Empty lleva los suyos: su suavizado, sus fps y de que archivo vino."""

    es_fuente: BoolProperty(default=False)
    archivo: StringProperty(name="Archivo", default="")
    info: StringProperty(name="info", default="")
    ruta_audio: StringProperty(name="Audio", default="", subtype='FILE_PATH')
    tira_sonido: StringProperty(default="")
    oir_audio: BoolProperty(
        name="Oir este audio",
        description="Silencia o deja sonar la pista de este audio en el secuenciador. "
                    "Al analizar, el archivo se mete siempre; esto solo decide si suena. "
                    "Activarlo enciende tambien el sonido al arrastrar el cursor",
        default=True, update=_al_cambiar_oir_audio,
    )

    fps: IntProperty(
        name="FPS del analisis",
        description="Los mismos --fps que usaste en analiza_audio.py para ESTE archivo",
        default=24, min=1, max=240,
        update=_al_mover_suavizado,  # el suavizado va en segundos: depende de los fps
    )
    ataque: FloatProperty(
        name="Ataque (s)",
        description="Segundos que tarda el valor en SUBIR de 0 a 1 como mucho. "
                    "0 = subida instantanea, que suele ser lo que quieres para que el "
                    "golpe entre seco. Subelo para dar sensacion de peso o inercia",
        default=0.0, min=0.0, soft_max=1.0, max=30.0, precision=2, step=5,
        update=_al_mover_suavizado,
    )
    caida: FloatProperty(
        name="Caida (s)",
        description="Segundos que tarda un pico en BAJAR de 1 a 0 como mucho. "
                    "0 = caida instantanea (sigue al audio tal cual). Mas segundos = cola "
                    "mas larga, como el release de un compresor",
        default=0.0, min=0.0, soft_max=2.0, max=30.0, precision=2, step=5,
        update=_al_mover_suavizado,
    )

    # ---- compas ----
    compas_activo: BoolProperty(default=False)
    bpm: FloatProperty(
        name="BPM",
        description="Pulsos por minuto. Lo pone el detector, pero puedes corregirlo a "
                    "mano si sabes el tempo exacto del tema",
        default=120.0, min=20.0, max=400.0, precision=2, step=10,
        update=_al_cambiar_compas,
    )
    desfase_compas: FloatProperty(
        name="Desfase (fotogramas)",
        description="Donde cae el primer pulso. Muevelo si el ritmo va adelantado o "
                    "atrasado respecto a la musica",
        default=0.0, min=-10000.0, max=10000.0, precision=2,
        update=_al_cambiar_compas,
    )
    pulsos_por_compas: IntProperty(
        name="Pulsos por compas",
        description="4 para un 4/4 normal, 3 para un vals",
        default=4, min=1, max=16, update=_al_cambiar_compas,
    )
    caida_pulso: FloatProperty(
        name="Caida del pulso (s)",
        description="Cuanto tarda 'pulso' en bajar de 1 a 0 tras cada golpe",
        default=0.25, min=0.01, soft_max=2.0, max=10.0, precision=2,
        update=_al_cambiar_compas,
    )
    banda_min_compas: IntProperty(
        name="Detectar desde", default=0, min=0, max=31,
        description="Bandas que se miran para encontrar el ritmo. Las graves suelen "
                    "llevar el bombo; si el tema no tiene bateria, prueba con todas",
    )
    banda_max_compas: IntProperty(name="hasta", default=3, min=0, max=31)
    bpm_detectado: FloatProperty(default=0.0)
    # 0..1. Normalmente es la fraccion de trozos del tema que coinciden en el
    # tempo; en temas cortos, que no se pueden partir, cae en la medida vieja.
    fiabilidad_compas: FloatProperty(default=0.0, min=0.0, max=1.0)
    # Los dos numeros de esa comprobacion, para poder decirlo con palabras
    # ("4 de 6 trozos coinciden") en vez de con un porcentaje abstracto.
    compas_acuerdo: IntProperty(default=0)
    compas_trozos: IntProperty(default=0)
    marcadores: BoolProperty(
        name="Marcadores en la linea de tiempo",
        description="Pone una marca al principio de cada compas, para verlo al hacer scrub",
        default=False, update=_al_cambiar_marcadores,
    )


class AV_Ajustes(PropertyGroup):
    """Ajustes de la escena: lo necesario para importar audios nuevos y para
    crear presets. El suavizado ya no vive aqui, sino en cada fuente."""

    fuente: PointerProperty(
        name="Fuente activa",
        description="Sobre que audio actuan el suavizado y los botones de crear presets",
        type=bpy.types.Object, poll=_poll_fuente,
    )

    ruta_json: StringProperty(
        name="Archivo JSON",
        description="Un .json ya analizado (por este mismo plugin o por analiza_audio.py)",
        subtype='FILE_PATH',
    )
    fotograma_inicial: IntProperty(
        name="Fotograma inicial",
        description="A que fotograma de Blender corresponde el fotograma 0 del audio",
        default=1,
    )
    fps: IntProperty(
        name="FPS",
        description="Fotogramas por segundo del analisis. Al analizar aqui, lo normal es "
                    "dejarlo igual que la escena. Al cargar un .json de fuera tiene que "
                    "coincidir con el --fps con el que se genero, o se ira desincronizando",
        default=24, min=1, max=240,
    )

    # ---- ajustes del analisis (solo al analizar dentro de Blender) ----
    mostrar_avanzado: BoolProperty(default=False)
    bandas: IntProperty(
        name="Bandas", description="En cuantas bandas se reparte el espectro",
        default=8, min=1, max=32,
    )
    ventana: EnumProperty(
        name="Ventana FFT",
        description="Muestras por FFT. Mas grande = mas detalle en los graves pero "
                    "transitorios mas emborronados. Es el mismo dilema que velocidad de "
                    "obturacion contra motion blur",
        items=(('512', "512", "Muy seca, poca resolucion en graves"),
               ('1024', "1024", "Seca"),
               ('2048', "2048", "Equilibrada (recomendada a 24 fps)"),
               ('4096', "4096", "Mas detalle en graves, transitorios blandos"),
               ('8192', "8192", "Solo para material muy grave y lento")),
        default='2048',
    )
    fmin: FloatProperty(name="Graves desde (Hz)", default=30.0, min=1.0, max=20000.0)
    fmax: FloatProperty(name="Agudos hasta (Hz)", default=16000.0, min=2.0, max=30000.0)
    rango_db: FloatProperty(
        name="Rango dinamico (dB)",
        description="Cuantos dB por debajo del pico se consideran 0. Menos = mas contraste",
        default=60.0, min=6.0, max=140.0,
    )
    canal_analisis: EnumProperty(
        name="Ver el canal",
        items=(('MONO', "Mono", "La mezcla"),
               ('IZQ', "Izquierdo", "Solo el canal izquierdo"),
               ('DER', "Derecho", "Solo el canal derecho")),
        default='MONO',
    )
    analizar_estereo: BoolProperty(
        name="Analizar en estereo",
        description="Analiza los dos canales por separado, ademas de la mezcla. Triplica "
                    "el tiempo de analisis y el tamano de los datos, asi que solo merece "
                    "la pena si vas a usarlo. Si el archivo es mono no hace nada",
        default=False,
    )
    norm: EnumProperty(
        name="Normalizar",
        items=(('BANDA', "Por banda", "Cada banda contra su propio pico: mas vistoso"),
               ('GLOBAL', "Global", "Un unico pico para todas: mas fiel, mandan los graves")),
        default='BANDA',
    )
    guardar_json: BoolProperty(
        name="Guardar tambien el .json",
        description="Deja el analisis en un .json junto al audio, para reutilizarlo sin "
                    "volver a analizar o pasarselo a otra persona",
        default=False,
    )
    ajustar_escena: BoolProperty(
        name="Ajustar la escena",
        description="Pone los fps y el rango de la linea de tiempo para que cuadren con el audio",
        default=True,
    )
    detectar_compas_al_cargar: BoolProperty(
        name="Detectar el compas al cargar",
        description="Busca el tempo nada mas analizar, sin tener que pulsar nada. "
                    "Cuesta unas decimas de segundo. Si el tema no tiene un ritmo "
                    "claro no pasa nada: lo dice y sigue",
        default=True,
    )

    altura: FloatProperty(
        name="Altura maxima",
        description="Cuanto mide la barra cuando la banda vale 1.0",
        default=4.0, min=0.01, soft_max=20.0, unit='LENGTH',
        update=_al_cambiar_drivers_barras,
    )
    base: FloatProperty(
        name="Altura minima",
        description="Cuanto mide la barra en silencio (para que no desaparezca)",
        default=0.05, min=0.0, soft_max=2.0, unit='LENGTH',
        update=_al_cambiar_drivers_barras,
    )
    ancho: FloatProperty(
        name="Ancho",
        default=0.8, min=0.01, soft_max=5.0, unit='LENGTH',
    )
    separacion: FloatProperty(
        name="Separacion",
        description="Distancia entre los centros de dos barras",
        default=1.0, min=0.01, soft_max=10.0, unit='LENGTH',
    )
    emision: BoolProperty(
        name="Material emisivo reactivo",
        description="Crea materiales de emision cuyo brillo tambien sigue a la banda",
        default=True,
    )
    fuerza_emision: FloatProperty(
        name="Brillo maximo",
        description="Emision cuando la banda vale 1.0. Por encima de ~5 el view "
                    "transform AgX de Blender lava el color hacia el blanco",
        default=3.0, min=0.0, soft_max=50.0,
    )
    barras_canal: EnumProperty(
        name="Canal", items=CANALES, default='MONO',
        description="En estereo se hacen el doble de barras, en espejo: los graves al "
                    "centro y los agudos a los extremos, el canal izquierdo a la izquierda",
        update=_al_cambiar_drivers_barras,
    )
    barras_pulso: FloatProperty(
        name="Golpe del compas",
        description="Cuanto sube cada barra de golpe en cada pulso, ademas de lo que "
                    "haga su banda. Necesita el compas detectado",
        default=0.0, min=0.0, soft_max=5.0, unit='LENGTH',
        update=_al_cambiar_drivers_barras,
    )

    # ---- preset 1: ecualizador de LEDs ----
    led_segmentos: IntProperty(
        name="Segmentos",
        description="Cubos por columna. Cuantos mas, mas fina la escala y menos retro",
        default=12, min=2, max=64,
    )
    led_ancho: FloatProperty(name="Ancho del cubo", default=0.7, min=0.01, soft_max=3.0, unit='LENGTH')
    led_alto: FloatProperty(name="Alto del cubo", default=0.22, min=0.01, soft_max=3.0, unit='LENGTH')
    led_paso: FloatProperty(
        name="Paso vertical",
        description="Distancia entre los centros de dos cubos de la misma columna. "
                    "Si es mayor que el alto, queda hueco entre ellos",
        default=0.34, min=0.01, soft_max=3.0, unit='LENGTH',
    )
    led_separacion: FloatProperty(name="Separacion de columnas", default=1.0, min=0.01,
                                  soft_max=10.0, unit='LENGTH')
    led_dureza: FloatProperty(
        name="Dureza",
        description="Como de brusco es el encendido. Alto = digital seco, bajo = el "
                    "segmento de arriba se enciende gradualmente",
        default=40.0, min=1.0, soft_max=100.0,
        update=_al_cambiar_drivers_led,
    )
    led_brillo: FloatProperty(
        name="Brillo encendido",
        description="Por encima de ~4 el view transform AgX lava el verde y el rojo "
                    "hacia el pastel y se pierde el aire de vumetro",
        default=2.5, min=0.0, soft_max=50.0,
    )
    led_apagado: FloatProperty(
        name="Brillo apagado",
        description="Los segmentos apagados no desaparecen: quedan tenues, como los "
                    "LEDs de verdad",
        default=0.05, min=0.0, soft_max=2.0,
    )
    led_canal: EnumProperty(
        name="Canal", items=CANALES, default='MONO',
        description="En estereo se hacen el doble de columnas, en espejo",
        update=_al_cambiar_drivers_led,
    )
    led_pulso: FloatProperty(
        name="Golpe del compas",
        description="En cada pulso se encienden segmentos de mas, como si la aguja "
                    "pegara un salto. Necesita el compas detectado",
        default=0.0, min=0.0, max=1.0,
        update=_al_cambiar_drivers_led,
    )

    # ---- preset: el compas en cubos ----
    pulso_lado: FloatProperty(name="Lado del cubo", default=0.8, min=0.01,
                              soft_max=5.0, unit='LENGTH',
                              update=_al_cambiar_pulso_vis)
    pulso_separacion: FloatProperty(name="Separacion", default=1.3, min=0.01,
                                    soft_max=10.0, unit='LENGTH',
                                    update=_al_cambiar_pulso_vis)
    pulso_crecimiento: FloatProperty(
        name="Cuanto crece",
        description="Cuanto se estira el cubo en su tiempo. 1.5 = un cubo y medio mas alto",
        default=1.5, min=0.0, soft_max=8.0, update=_al_cambiar_pulso_vis,
    )
    pulso_brillo: FloatProperty(name="Brillo en su tiempo", default=6.0, min=0.0,
                                soft_max=40.0, update=_al_cambiar_pulso_vis)
    pulso_apagado: FloatProperty(
        name="Brillo en reposo",
        description="Los cubos que no tocan no desaparecen: se quedan tenues",
        default=0.4, min=0.0, soft_max=5.0, update=_al_cambiar_pulso_vis,
    )


class AV_PaisajeAjustes(PropertyGroup):
    """Configuracion de UN paisaje. Va pegada al objeto, como la del plexus."""

    es_paisaje: BoolProperty(default=False)
    fuente: PointerProperty(
        name="Audio",
        description="Que audio dibuja el relieve",
        type=bpy.types.Object, poll=_poll_fuente, update=_al_cambiar_paisaje,
    )

    filas: IntProperty(
        name="Filas (tiempo)",
        description="Cuantos instantes del pasado se ven a la vez. Mas filas = horizonte "
                    "mas profundo y malla mas densa",
        default=64, min=2, soft_max=160, max=512, update=_al_cambiar_paisaje,
    )
    columnas: IntProperty(
        name="Columnas por copia",
        description="Resolucion a lo ancho de UNA copia. Las bandas se interpolan entre "
                    "columnas",
        default=64, min=2, soft_max=160, max=512, update=_al_cambiar_paisaje,
    )
    repeticiones: IntProperty(
        name="Copias a lo ancho",
        description="Repite el paisaje a los lados para llenar mas pantalla. Cada copia "
                    "mide lo que diga 'Ancho', asi que el total crece con ellas",
        default=1, min=1, max=12, update=_al_cambiar_paisaje,
    )
    espejo: BoolProperty(
        name="Copias en espejo",
        description="Alterna el sentido de cada copia -grave a agudo, agudo a grave- para "
                    "que en la union coincidan y no se vea la costura",
        default=True, update=_al_cambiar_paisaje,
    )
    ancho: FloatProperty(name="Ancho", default=12.0, min=0.01, soft_max=100.0,
                         unit='LENGTH', update=_al_cambiar_paisaje)
    largo: FloatProperty(name="Fondo", default=16.0, min=0.01, soft_max=200.0,
                         unit='LENGTH', update=_al_cambiar_paisaje)
    altura: FloatProperty(
        name="Altura de las montanas",
        description="Cuanto se levanta el relieve cuando la banda vale 1.0",
        default=2.5, min=0.0, soft_max=30.0, unit='LENGTH', update=_al_cambiar_paisaje,
    )
    direccion: EnumProperty(name="Avanza", items=DIRECCIONES, default='SUR',
                            update=_al_cambiar_paisaje)
    fotogramas_por_fila: FloatProperty(
        name="Fotogramas por fila",
        description="Cuanto tiempo cubre cada fila. Con 1 el relieve avanza una fila por "
                    "fotograma; con 2 avanza media (mas lento, pero se ve el doble de "
                    "historia); con 0.5 avanza dos (mas rapido y con mas detalle)",
        default=1.0, min=0.05, soft_max=6.0, max=60.0, precision=2,
        update=_al_cambiar_paisaje,
    )
    canal: EnumProperty(
        name="Canal", items=CANALES, default='MONO',
        description="En estereo, el lado izquierdo del terreno sigue al canal izquierdo "
                    "y el derecho al derecho",
        update=_al_cambiar_paisaje,
    )
    banda_min: IntProperty(name="De la banda", default=0, min=0, max=31,
                           update=_al_cambiar_paisaje)
    banda_max: IntProperty(name="a la", default=7, min=0, max=31,
                           update=_al_cambiar_paisaje)
    suave: BoolProperty(
        name="Interpolar entre bandas",
        description="Relieve continuo. Desactivalo para que salgan crestas rectas, una "
                    "por banda, en plan ecualizador",
        default=True, update=_al_cambiar_paisaje,
    )

    # ---- moldear el relieve ----
    ganancia: FloatProperty(
        name="Ganancia",
        description="Multiplica toda la altura. Por encima de 1 las cumbres se aplanan "
                    "contra el techo, que tambien es un efecto util",
        default=1.0, min=0.0, soft_max=4.0, max=20.0, update=_al_cambiar_paisaje,
    )
    curva: FloatProperty(
        name="Curva",
        description="Reparte el relieve. Por debajo de 1 levanta los valles y sale "
                    "terreno por todas partes; por encima de 1 aplasta todo menos los "
                    "picos y quedan montanas sueltas sobre un llano",
        default=1.0, min=0.05, soft_max=4.0, max=10.0, update=_al_cambiar_paisaje,
    )
    inclinacion: FloatProperty(
        name="Balance graves/agudos",
        description="Una balanza entre los dos extremos del espectro, como un ecualizador "
                    "de inclinacion. Hacia -1 mandan los graves; hacia +1 los agudos, que "
                    "de natural son mucho mas debiles",
        default=0.0, min=-1.0, max=1.0, update=_al_cambiar_paisaje,
    )
    suelo: FloatProperty(
        name="Suelo",
        description="Levanta el terreno entero para que los valles no queden planos",
        default=0.0, min=0.0, max=1.0, update=_al_cambiar_paisaje,
    )
    pulso_altura: FloatProperty(
        name="Cresta del compas",
        description="Cada golpe levanta la fila que le corresponde, y como las filas "
                    "avanzan, esa cresta viaja por el terreno: se ve el ritmo alejandose "
                    "hacia el horizonte. Necesita el compas detectado",
        default=0.0, min=0.0, max=1.0, update=_al_cambiar_paisaje,
    )
    pulso_extension: FloatProperty(
        name="Ancho de la cresta",
        description="Cuanto del espectro abarca. Con 1 es una cresta recta de lado a "
                    "lado, que parece un escalon; bajandolo se concentra en los graves y "
                    "queda una loma que nace donde esta el bombo",
        default=0.35, min=0.02, max=1.0, update=_al_cambiar_paisaje,
    )
    compas_marca: FloatProperty(
        name="Marca de compas",
        description="Una muesca en el PRIMER tiempo de cada compas, pegada a una orilla. "
                    "Con la cresta sola todos los tiempos son iguales y no se ve donde "
                    "empieza cada compas; esto deja un carril que hace de regla",
        default=0.0, min=0.0, max=1.0, update=_al_cambiar_paisaje,
    )
    compas_marca_lado: EnumProperty(
        name="En el lado de",
        description="En que orilla va la marca. Por defecto en los agudos, porque la "
                    "cresta del pulso se concentra en los graves y se taparian",
        items=(('AGUDOS', "Los agudos", "Al borde de las frecuencias altas"),
               ('GRAVES', "Los graves", "Al borde de las frecuencias bajas")),
        default='AGUDOS', update=_al_cambiar_paisaje,
    )
    compas_marca_ancho: FloatProperty(
        name="Ancho de la marca",
        description="Cuanto se mete hacia dentro desde la orilla",
        default=0.15, min=0.02, max=1.0, update=_al_cambiar_paisaje,
    )

    modo: EnumProperty(name="Aspecto", items=MODOS_PAISAJE, default='AMBOS',
                       update=_al_cambiar_estilo_paisaje)
    grosor_malla: FloatProperty(
        name="Grosor de los hilos",
        description="En fracciones de celda: 0.5 seria una celda entera pintada",
        default=0.08, min=0.001, max=0.5, precision=3,
        update=_al_cambiar_estilo_paisaje,
    )
    opacidad_superficie: FloatProperty(
        name="Opacidad de la superficie",
        description="Con 'Solido + malla', cuanto se ve la superficie entre los hilos",
        default=0.3, min=0.0, max=1.0, update=_al_cambiar_estilo_paisaje,
    )

    color_bajo: FloatVectorProperty(name="Valles", subtype='COLOR', size=3,
                                    default=(0.02, 0.1, 0.4), min=0.0, max=1.0,
                                    update=_al_cambiar_color_paisaje)
    color_alto: FloatVectorProperty(name="Cumbres", subtype='COLOR', size=3,
                                    default=(1.0, 0.35, 0.75), min=0.0, max=1.0,
                                    update=_al_cambiar_color_paisaje)
    brillo: FloatProperty(name="Brillo", default=2.0, min=0.0, soft_max=20.0,
                          update=_al_cambiar_color_paisaje)
    desvanecer: BoolProperty(
        name="Desvanecer el horizonte",
        description="Difumina el fondo del paisaje. Sin esto, el relieve se corta en seco "
                    "con una linea recta que delata que aquello es una rejilla",
        default=True, update=_al_cambiar_color_paisaje,
    )
    desvanecido: FloatProperty(
        name="Cuanto se desvanece",
        description="Que parte del fondo se difumina, de 0 a 1",
        default=0.4, min=0.0, max=1.0, update=_al_cambiar_color_paisaje,
    )


class AV_EnjambreAjustes(PropertyGroup):
    """Configuracion de UN enjambre, pegada al objeto como la del plexus."""

    es_enjambre: BoolProperty(default=False)

    fuente: PointerProperty(
        name="Audio",
        description="Que fuente de audio mueve este enjambre",
        type=bpy.types.Object, poll=_poll_fuente, update=_al_cambiar_enjambre,
    )

    particulas: IntProperty(
        name="Particulas",
        description="Cuantas hay. Unas miles se distinguen de una en una; a partir "
                    "de decenas de miles se lee como polvo, y el visor va mas lento",
        default=5000, min=8, soft_max=20000, max=200000, update=_al_cambiar_enjambre,
    )
    forma: EnumProperty(name="Forma", items=FORMAS_ENJAMBRE, default='DISCO',
                        update=_al_cambiar_enjambre)
    reparto: EnumProperty(name="Bandas", items=REPARTOS_ENJAMBRE, default='RADIO',
                          update=_al_cambiar_enjambre)
    radio: FloatProperty(name="Radio", default=5.0, min=0.01, soft_max=40.0,
                         unit='LENGTH', update=_al_cambiar_enjambre)
    grosor: FloatProperty(
        name="Grosor",
        description="Lo que se sale la nube de su plano. A 0 queda un disco "
                    "perfectamente plano; subiendolo engorda hacia una lenteja",
        default=0.15, min=0.0, soft_max=2.0, update=_al_cambiar_enjambre,
    )
    semilla: IntProperty(name="Semilla", default=0, min=0, max=9999,
                         update=_al_cambiar_enjambre)

    giro: FloatProperty(
        name="Giro",
        description="Vueltas por segundo. En negativo gira al reves",
        default=0.05, min=-5.0, max=5.0, update=_al_cambiar_enjambre,
    )
    diferencial: FloatProperty(
        name="Giro diferencial",
        description="Cuanto mas rapido van las de dentro que las de fuera. A 0 la "
                    "nube gira como un solido, que se lee como algo rigido; subido "
                    "se enrosca sola y parece una galaxia",
        default=1.0, min=0.0, soft_max=6.0, update=_al_cambiar_enjambre,
    )
    empuje: FloatProperty(
        name="Empuje directo",
        description="Coloca la particula segun lo que suena AHORA: si el audio "
                    "pega un salto, ella pega el salto. Es el mas directo de "
                    "leer, pero con musica movida tiembla",
        default=1.2, min=0.0, soft_max=20.0, unit='LENGTH',
        update=_al_cambiar_enjambre,
    )
    fuerza: FloatProperty(
        name="Fuerza con inercia",
        description="Trata cada golpe como un empujon sobre algo que pesa: sale "
                    "despacio y vuelve solo. Mucho mas sutil. Es lo que se "
                    "desplaza una particula con su banda sonando a tope y "
                    "sostenida. Se puede mezclar con el empuje directo, o "
                    "usarlo solo bajando aquel a cero",
        default=0.0, min=0.0, soft_max=20.0, unit='LENGTH',
        update=_al_cambiar_enjambre,
    )
    vuelta: FloatProperty(
        name="Tiempo de vuelta",
        description="Lo que tarda en regresar tras un empujon. Corto = nervioso "
                    "y pegado al ritmo; largo = pesado, las particulas siguen "
                    "moviendose cuando el golpe ya paso",
        default=0.6, min=0.02, soft_max=5.0, subtype='TIME_ABSOLUTE', unit='TIME',
        update=_al_cambiar_enjambre,
    )
    rebote: FloatProperty(
        name="Rebote",
        description="A 0 vuelve sin pasarse, como con el freno puesto. Subiendolo "
                    "se pasa de largo y oscila unas cuantas veces, como un muelle",
        default=0.35, min=0.0, max=1.0, update=_al_cambiar_enjambre,
    )

    peso_graves: FloatProperty(
        name="Graves",
        description="Cuanto mandan los graves sobre el movimiento Y sobre el brillo. "
                    "A 0 las particulas graves ni se mueven ni se encienden",
        default=1.0, min=0.0, soft_max=3.0, update=_al_cambiar_enjambre,
    )
    peso_medios: FloatProperty(
        name="Medios", description="Lo mismo para la zona central del espectro",
        default=1.0, min=0.0, soft_max=3.0, update=_al_cambiar_enjambre,
    )
    peso_agudos: FloatProperty(
        name="Agudos", description="Lo mismo para los agudos",
        default=1.0, min=0.0, soft_max=3.0, update=_al_cambiar_enjambre,
    )

    turbulencia: FloatProperty(
        name="Turbulencia",
        description="Una corriente de remolinos que arrastra la nube. Quita el aire "
                    "de rejilla perfecta",
        default=0.0, min=0.0, soft_max=10.0, unit='LENGTH',
        update=_al_cambiar_enjambre,
    )
    turb_audio: FloatProperty(
        name="La agita el audio",
        description="A 0 la corriente sopla siempre igual. Subiendolo, solo se "
                    "agitan las particulas cuya banda esta sonando: en los golpes "
                    "fuertes hierve y en los silencios se queda quieta. A 1 la "
                    "turbulencia depende por completo de la musica",
        default=0.0, min=0.0, max=1.0, update=_al_cambiar_enjambre,
    )
    turb_escala: FloatProperty(
        name="Tamano del remolino",
        description="Grande = zonas enteras de la nube se mueven juntas, como una "
                    "corriente. Pequeno = cada particula por su lado, y vuelve el "
                    "temblor",
        default=3.0, min=0.05, soft_max=30.0, unit='LENGTH',
        update=_al_cambiar_enjambre,
    )
    turb_velocidad: FloatProperty(
        name="Velocidad del remolino",
        description="Lo rapido que cambia la corriente. Bajo se percibe como algo "
                    "vivo pero tranquilo",
        default=0.3, min=0.0, soft_max=5.0, update=_al_cambiar_enjambre,
    )

    pulso_onda: FloatProperty(
        name="Onda por pulso",
        description="Un anillo que sale del centro en cada golpe. Ademas de empujar, "
                    "enciende las particulas por donde pasa. Necesita el compas detectado",
        default=0.0, min=0.0, soft_max=10.0, unit='LENGTH', update=_al_cambiar_enjambre,
    )
    onda_grosor: FloatProperty(name="Grosor", default=0.18, min=0.01, max=1.0,
                               update=_al_cambiar_enjambre)
    compas_onda: FloatProperty(
        name="Onda por compas",
        description="La segunda capa, mas lenta: tarda un compas entero en cruzar",
        default=0.0, min=0.0, soft_max=10.0, unit='LENGTH', update=_al_cambiar_enjambre,
    )
    compas_onda_grosor: FloatProperty(name="Grosor", default=0.30, min=0.01, max=1.0,
                                      update=_al_cambiar_enjambre)

    canal: EnumProperty(name="Canal", items=CANALES, default='MONO',
                        update=_al_cambiar_enjambre)
    suave: BoolProperty(name="Interpolar entre bandas", default=True,
                        update=_al_cambiar_enjambre)
    banda_min: IntProperty(name="Banda desde", default=0, min=0, max=31,
                           update=_al_cambiar_enjambre)
    banda_max: IntProperty(name="Banda hasta", default=7, min=0, max=31,
                           update=_al_cambiar_enjambre)

    tam_punto: FloatProperty(name="Tamano", default=0.03, min=0.0001, soft_max=0.5,
                             precision=4, unit='LENGTH',
                             update=_al_cambiar_estilo_enjambre)
    reaccion_tam: FloatProperty(
        name="Crecer al sonar",
        description="Cuanto engorda una particula cuando su banda suena. A 0 todas "
                    "miden igual y solo cambia el brillo",
        default=2.0, min=0.0, soft_max=10.0, update=_al_cambiar_estilo_enjambre,
    )

    color_grave: FloatVectorProperty(name="Graves", subtype='COLOR', size=3,
                                     default=(0.0, 0.45, 1.0), min=0.0, max=1.0,
                                     update=_al_cambiar_color_enjambre)
    color_medio: FloatVectorProperty(name="Medios", subtype='COLOR', size=3,
                                     default=(0.1, 1.0, 0.9), min=0.0, max=1.0,
                                     update=_al_cambiar_color_enjambre)
    color_agudo: FloatVectorProperty(name="Agudos", subtype='COLOR', size=3,
                                     default=(1.0, 0.15, 0.6), min=0.0, max=1.0,
                                     update=_al_cambiar_color_enjambre)
    brillo: FloatProperty(name="Brillo", default=2.5, min=0.0, soft_max=20.0,
                          update=_al_cambiar_color_enjambre)
    fondo: FloatProperty(
        name="Suelo de brillo",
        description="Lo que iluminan las particulas calladas. A 0 desaparecen y "
                    "solo se ve lo que suena; subido se ve siempre la nube entera",
        default=0.08, min=0.0, max=1.0, update=_al_cambiar_color_enjambre,
    )
    destello: FloatProperty(
        name="Destello de la onda",
        description="Cuanto encienden las ondas al pasar, aparte de empujar",
        default=1.0, min=0.0, soft_max=10.0, update=_al_cambiar_color_enjambre,
    )


class AV_PlexusAjustes(PropertyGroup):
    """Configuracion de UN plexus. Se registra en bpy.types.Object, asi que cada
    objeto lleva la suya, se guarda con el .blend y se copia sola al duplicar."""

    es_plexus: BoolProperty(default=False)

    fuente: PointerProperty(
        name="Audio",
        description="Que fuente de audio mueve este plexus. Si tienes varias cargadas, "
                    "cada plexus puede escuchar una distinta",
        type=bpy.types.Object, poll=_poll_fuente, update=_al_cambiar_plexus,
    )

    forma: EnumProperty(name="Forma", items=FORMAS, default='ESFERA',
                        update=_al_cambiar_origen)
    objeto_origen: PointerProperty(
        name="Modelo",
        description="El objeto de la escena del que se sacan los puntos. Se lee con sus "
                    "modificadores aplicados, asi que puedes usar Subdivision, Remesh...",
        type=bpy.types.Object, poll=_poll_modelo, update=_al_cambiar_origen,
    )
    puntos: IntProperty(
        name="Puntos",
        description="Densidad de la nube. Por encima de ~1200 la reconstruccion por "
                    "fotograma empieza a notarse al reproducir",
        default=250, min=4, soft_max=1200, max=4000, update=_al_cambiar_plexus,
    )
    radio: FloatProperty(name="Radio", default=4.0, min=0.01, soft_max=30.0,
                         unit='LENGTH', update=_al_cambiar_plexus)
    semilla: IntProperty(name="Semilla", description="Cambiala para obtener otro reparto",
                         default=0, min=0, max=9999, update=_al_cambiar_plexus)
    amplitud: FloatProperty(
        name="Amplitud",
        description="Cuanto se desplaza un punto cuando su banda vale 1.0",
        default=1.5, min=0.0, soft_max=20.0, unit='LENGTH', update=_al_cambiar_plexus,
    )
    pulso_amplitud: FloatProperty(
        name="Golpe (toda la nube)",
        description="En cada pulso la nube entera pega un empujon hacia fuera. Es un "
                    "latido uniforme: se nota, pero parece mas que el objeto engorda "
                    "que un ritmo. Necesita el compas detectado",
        default=0.0, min=0.0, soft_max=10.0, unit='LENGTH', update=_al_cambiar_plexus,
    )
    pulso_onda: FloatProperty(
        name="Onda por pulso",
        description="Un anillo que sale del centro en cada golpe y llega al borde justo "
                    "cuando entra el siguiente. Se lee como ritmo mucho mejor que el "
                    "latido uniforme. Necesita el compas detectado",
        default=0.0, min=0.0, soft_max=10.0, unit='LENGTH', update=_al_cambiar_plexus,
    )
    onda_grosor: FloatProperty(
        name="Grosor",
        description="Estrecho = un latigazo que recorre la nube; ancho = una marea",
        default=0.18, min=0.01, max=1.0, update=_al_cambiar_plexus,
    )
    compas_onda: FloatProperty(
        name="Onda por compas",
        description="Otro anillo, pero este tarda un compas entero en cruzar. Puesto a la "
                    "vez que el del pulso da dos capas de ritmo: una rapida y una de "
                    "fondo que marca donde empieza cada compas",
        default=0.0, min=0.0, soft_max=10.0, unit='LENGTH', update=_al_cambiar_plexus,
    )
    compas_onda_grosor: FloatProperty(
        name="Grosor",
        description="Suele quedar mejor mas ancho que el del pulso: al ir tan despacio, "
                    "un anillo fino se queda en un hilo que apenas se ve moverse",
        default=0.30, min=0.01, max=1.0, update=_al_cambiar_plexus,
    )
    canal: EnumProperty(
        name="Canal", items=CANALES, default='MONO',
        description="En estereo, la mitad izquierda de la nube sigue al canal izquierdo "
                    "y la derecha al derecho, fundiendose por el centro",
        update=_al_cambiar_plexus,
    )
    asignacion: EnumProperty(name="Bandas por", items=ASIGNACIONES, default='VERTICAL',
                             update=_al_cambiar_plexus)
    suave: BoolProperty(
        name="Interpolar entre bandas",
        description="Cada punto mezcla las dos bandas mas cercanas y la superficie queda "
                    "continua. Desactivalo para que se formen mesetas y escalones, como "
                    "un ecualizador clasico",
        default=True, update=_al_cambiar_plexus,
    )
    banda_min: IntProperty(name="Banda desde", default=0, min=0, max=31,
                           update=_al_cambiar_plexus)
    banda_max: IntProperty(name="Banda hasta", default=7, min=0, max=31,
                           update=_al_cambiar_plexus)
    distancia: FloatProperty(
        name="Distancia de union",
        description="Dos puntos se unen si estan mas cerca que esto. Es el control que "
                    "mas cambia el aspecto: bajo = puntos sueltos, alto = maraña",
        default=1.6, min=0.0, soft_max=20.0, unit='LENGTH', update=_al_cambiar_plexus,
    )
    conexiones: IntProperty(
        name="Conexiones por punto",
        description="Tope de vecinos a los que se une cada punto, para que las zonas "
                    "densas no se conviertan en una mancha",
        default=4, min=1, max=32, update=_al_cambiar_plexus,
    )
    grosor: FloatProperty(name="Grosor de linea", default=0.012, min=0.0001,
                          soft_max=0.2, precision=4, unit='LENGTH',
                          update=_al_cambiar_estilo_plexus)
    tam_punto: FloatProperty(name="Tamano del punto", default=0.05, min=0.0001,
                             soft_max=0.5, precision=4, unit='LENGTH',
                             update=_al_cambiar_estilo_plexus)

    color_grave: FloatVectorProperty(name="Graves", subtype='COLOR', size=3,
                                     default=(0.0, 0.45, 1.0), min=0.0, max=1.0,
                                     update=_al_cambiar_color_plexus)
    color_medio: FloatVectorProperty(name="Medios", subtype='COLOR', size=3,
                                     default=(0.1, 1.0, 0.9), min=0.0, max=1.0,
                                     update=_al_cambiar_color_plexus)
    color_agudo: FloatVectorProperty(name="Agudos", subtype='COLOR', size=3,
                                     default=(1.0, 0.15, 0.6), min=0.0, max=1.0,
                                     update=_al_cambiar_color_plexus)
    brillo: FloatProperty(name="Brillo", default=2.5, min=0.0, soft_max=20.0,
                          update=_al_cambiar_color_plexus)

    # ---- caras, en un objeto aparte ----
    caras: BoolProperty(
        name="Generar caras",
        description="Rellena con triangulos los huecos donde tres puntos estan unidos "
                    "entre si. Van a un objeto separado, hijo de este, para que puedas "
                    "darles su propio material sin tocar el de las lineas",
        default=False, update=_al_cambiar_caras,
    )
    objeto_caras: PointerProperty(type=bpy.types.Object)
    ratio_caras: FloatProperty(
        name="Ratio de aparicion",
        description="Que fraccion de los triangulos posibles se rellena. 1 = todos "
                    "(membrana cerrada), 0.3 = parches sueltos. El sorteo es estable: "
                    "un triangulo concreto sale o no sale siempre igual, no parpadea",
        default=1.0, min=0.0, max=1.0, update=_al_cambiar_plexus,
    )
    color_caras: FloatVectorProperty(name="En silencio", subtype='COLOR', size=3,
                                     default=(0.05, 0.25, 0.7), min=0.0, max=1.0,
                                     update=_al_cambiar_color_caras)
    degradado_caras: BoolProperty(
        name="Colorear por intensidad",
        description="La cara toma color segun lo que suenen sus vertices, interpolado. "
                    "Apagado, queda de un color plano",
        default=True, update=_al_cambiar_color_caras,
    )
    color_caras_alta: FloatVectorProperty(name="A tope", subtype='COLOR', size=3,
                                          default=(0.4, 0.95, 1.0), min=0.0, max=1.0,
                                          update=_al_cambiar_color_caras)
    atributo_caras: EnumProperty(
        name="Segun",
        description="Que atributo alimenta el degradado de las caras",
        items=(
            ('av_intensidad', "Intensidad", "Cuanto suena la banda de cada vertice ahora mismo"),
            ('av_nivel', "Banda", "Donde esta cada vertice en el espectro: grave o agudo"),
        ),
        default='av_intensidad', update=_al_cambiar_color_caras,
    )
    opacidad_caras: FloatProperty(name="Opacidad", default=0.18, min=0.0, max=1.0,
                                  update=_al_cambiar_color_caras)
    brillo_caras: FloatProperty(name="Brillo de las caras", default=1.0, min=0.0,
                                soft_max=20.0, update=_al_cambiar_color_caras)


# ---------------------------------------------------------------------------
# OPERADORES (los botones)
# ---------------------------------------------------------------------------

def instalar_fuente(escena, empty, fotogramas, matrices, n_bandas, nombre_corto, fps):
    """Deja los datos dentro del Empty y genera su animacion. Comun a las dos
    formas de cargar: analizando aqui o leyendo un .json de fuera.

    `matrices` es {'MONO': [...]} y, si el audio venia en estereo, tambien
    'IZQ' y 'DER'.
    """
    aud = empty.audioviz_audio
    aud.es_fuente = True
    aud.archivo = nombre_corto
    aud.fps = fps

    # Bandas de una carga anterior que tuviera mas, o que fuera estereo cuando
    # esta es mono.
    for clave in [k for k in empty.keys()
                  if k.startswith((PREFIJO_BANDA, PREFIJO_BANDA_IZQ, PREFIJO_BANDA_DER))]:
        del empty[clave]

    for canal in matrices:
        prefijo = PREFIJOS_CANAL[canal]
        etiqueta = {'MONO': "", 'IZQ': " izquierdo", 'DER': " derecho"}[canal]
        for i in range(n_bandas):
            clave = f"{prefijo}{i}"
            empty[clave] = 0.0
            empty.id_properties_ui(clave).update(
                min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, default=0.0,
                description=f"Banda {i}{etiqueta} (grave -> agudo)",
            )

    # Accion limpia: si el audio anterior tenia mas bandas o mas fotogramas,
    # quedarian curvas huerfanas.
    if empty.animation_data is not None and empty.animation_data.action is not None:
        vieja = empty.animation_data.action
        empty.animation_data.action = None
        if vieja.users == 0:
            bpy.data.actions.remove(vieja)

    guardar_crudos(empty, fotogramas, matrices, n_bandas)
    aplicar_suavizado(empty, aud.ataque, aud.caida, aud.fps)
    if aud.compas_activo:
        generar_compas(empty, escena)


def ajustar_rango_escena(escena, fotogramas, fps):
    inicios = [fotogramas[0]]
    finales = [fotogramas[-1]]
    for otra in fuentes_de_la_escena(escena):
        marcos = otra.get(CLAVE_FRAMES)
        if marcos:
            inicios.append(int(marcos[0]))
            finales.append(int(marcos[-1]))
    escena.frame_start = min(inicios)
    escena.frame_end = max(finales)
    escena.render.fps = int(round(fps))
    escena.render.fps_base = 1.0


class AV_OT_analizar_audio(Operator):
    bl_idname = "audioviz.analizar_audio"
    bl_label = "Analizar un audio"
    bl_description = ("Abre un archivo de audio, lo analiza aqui mismo y crea la fuente. "
                      "No necesita ningun programa de fuera: usa el decodificador y el "
                      "ffmpeg que Blender ya trae")
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(
        default=";".join("*" + e for e in EXTENSIONES_AUDIO), options={'HIDDEN'})
    reemplazar: BoolProperty(default=False, options={'HIDDEN'})

    def invoke(self, contexto, evento):
        contexto.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, contexto):
        if np is None:
            self.report({'ERROR'}, "Hace falta numpy y no esta disponible")
            return {'CANCELLED'}
        escena = contexto.scene
        aj = escena.audioviz

        ruta = bpy.path.abspath(self.filepath)
        if not ruta or not os.path.isfile(ruta):
            self.report({'ERROR'}, "Elige un archivo de audio")
            return {'CANCELLED'}

        wm = contexto.window_manager
        wm.progress_begin(0, 100)
        try:
            try:
                x, izq, der, frec = leer_audio(ruta)
            except Exception as e:
                self.report({'ERROR'}, f"No he podido abrir el audio: {e}")
                return {'CANCELLED'}

            if len(x) < 64:
                self.report({'ERROR'}, "El archivo es demasiado corto")
                return {'CANCELLED'}

            # El limite de Nyquist manda: por encima de la mitad de la frecuencia
            # de muestreo no hay informacion que analizar.
            f_max = min(aj.fmax, frec / 2.0 * 0.98)
            if f_max <= aj.fmin:
                self.report({'ERROR'}, "'Agudos hasta' tiene que ser mayor que 'Graves desde'")
                return {'CANCELLED'}

            estereo = izq is not None and aj.analizar_estereo
            pasos = 3 if estereo else 1
            hechas = [0]

            def avance(hechos, total, _p=pasos):
                frac = (hechas[0] + hechos / max(total, 1)) / _p
                wm.progress_update(int(frac * 100))

            def analiza(senal):
                d, b = analizar_muestras(senal, frec, aj.fps, aj.bandas,
                                         int(aj.ventana), aj.fmin, f_max,
                                         progreso=avance)
                hechas[0] += 1
                return d, b

            db, bordes = analiza(x)
            y = normalizar_db(db, aj.rango_db, aj.norm)
            matrices = {'MONO': y.tolist()}

            if estereo:
                # Los dos canales se analizan aparte y se normalizan con el
                # mismo techo, o la imagen estereo se perderia.
                db_i, _ = analiza(izq)
                db_d, _ = analiza(der)
                y_i, y_d = normalizar_estereo(db_i, db_d, aj.rango_db, aj.norm)
                matrices['IZQ'] = y_i.tolist()
                matrices['DER'] = y_d.tolist()
        finally:
            wm.progress_end()

        nombre_corto = os.path.splitext(os.path.basename(ruta))[0]
        col = obtener_coleccion(escena)

        if self.reemplazar:
            empty = fuente_activa(escena)
            if empty is None:
                self.report({'ERROR'}, "No hay ninguna fuente activa que reemplazar")
                return {'CANCELLED'}
        else:
            empty = bpy.data.objects.new(f"{PREFIJO_FUENTE}_{nombre_corto}", None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.3
            col.objects.link(empty)

        if empty.name not in col.objects:
            for c in list(empty.users_collection):
                c.objects.unlink(empty)
            col.objects.link(empty)

        fotogramas = [i + aj.fotograma_inicial for i in range(len(y))]
        instalar_fuente(escena, empty, fotogramas, matrices, y.shape[1],
                        nombre_corto, aj.fps)
        empty.audioviz_audio.ruta_audio = ruta

        if aj.ajustar_escena:
            ajustar_rango_escena(escena, fotogramas, aj.fps)

        # El audio se mete SIEMPRE. Que suene o no lo decide la casilla 'Oir este
        # audio' de la fuente, que es un mute de verdad y se puede cambiar cuando
        # quieras sin volver a cargar el archivo.
        # Va despues de ajustar los fps de la escena: la duracion de la tira se
        # calcula con ellos, y al reves saldria descuadrada.
        sonido = ""
        try:
            tira = poner_tira_sonido(escena, empty, ruta, fotogramas[0])
            sonido = (" · audio en el secuenciador"
                      if not tira.mute else " · audio en el secuenciador (silenciado)")
        except Exception as e:
            sonido = f" · no he podido meterlo en el secuenciador: {e}"

        extra = ""
        if aj.guardar_json:
            destino = Path(ruta).with_suffix(".json")
            datos = {str(n): [round(float(v), 4) for v in fila] for n, fila in enumerate(y)}
            try:
                destino.write_text(json.dumps(datos, separators=(",", ":")), encoding="utf-8")
                extra = f" · json en {destino.name}"
            except Exception as e:
                extra = f" · no he podido escribir el json: {e}"

        empty.audioviz_audio.info = (
            f"{y.shape[1]} bandas · {len(y)} fotogramas · {fotogramas[0]}-"
            f"{fotogramas[-1]} · {len(y) / max(aj.fps, 1):.1f}s · {frec} Hz")
        aj.fuente = empty

        # El compas, de una vez. Cuesta unas decimas de segundo -medido: 176 ms
        # en un tema de cuatro minutos, sobre los segundos que ya ha costado el
        # analisis- y casi siempre se acaba queriendo. Si sale mal, el boton de
        # la seccion del compas lo repite, y ahi se puede corregir a mano.
        compas = ""
        if aj.detectar_compas_al_cargar:
            bien, mensaje = detectar_compas_en(empty, escena)
            compas = f" · {mensaje}" if bien else f" · sin compas: {mensaje}"

        verbo = "Reemplazado" if self.reemplazar else "Analizado"
        self.report({'INFO'}, f"{verbo} '{os.path.basename(ruta)}': {len(y)} fotogramas, "
                              f"{bordes[0]:.0f}-{bordes[-1]:.0f} Hz{extra}{sonido}{compas}")
        return {'FINISHED'}


class AV_OT_ver_analisis(Operator):
    bl_idname = "audioviz.ver_analisis"
    bl_label = "Ver el analisis"
    bl_description = ("Dibuja el tema entero como espectrograma: el tiempo a lo ancho, "
                      "las bandas de grave a agudo, y los pulsos del compas encima. "
                      "Refleja el suavizado, asi que sirve para ajustarlo mirando")
    bl_options = {'REGISTER'}

    def execute(self, contexto):
        fuente = fuente_activa(contexto.scene)
        if fuente is None:
            self.report({'ERROR'}, "Carga primero un audio")
            return {'CANCELLED'}

        aj = contexto.scene.audioviz
        img, aviso = generar_espectro(fuente, aj.canal_analisis)
        if img is None:
            self.report({'ERROR'}, f"No he podido dibujarlo: {aviso}")
            return {'CANCELLED'}

        # Empaquetada, para que viaje dentro del .blend.
        try:
            img.pack()
        except Exception:
            pass

        # La miniatura del panel. Se pide AQUI y no al dibujar el panel: crear
        # cosas mientras Blender esta pintando la interfaz es justo lo que no
        # hay que hacer. En el panel solo se lee lo que ya existe.
        try:
            img.preview_ensure()
        except Exception:
            pass

        actualizar_marca_espectro(contexto.scene)

        self.report({'INFO'}, f"'{img.name}' listo ({img.size[0]}x{img.size[1]}). "
                              "Abrelo en un editor de imagenes para verlo grande")
        return {'FINISHED'}


class AV_OT_quitar_analisis(Operator):
    bl_idname = "audioviz.quitar_analisis"
    bl_label = "Quitar la vista"
    bl_description = "Borra la imagen del analisis"
    bl_options = {'REGISTER'}

    def execute(self, contexto):
        fuente = fuente_activa(contexto.scene)
        if fuente is None:
            return {'CANCELLED'}
        nombre = f"{NOMBRE_IMAGEN}_{etiqueta_fuente(fuente)}"
        _espectros.pop(nombre, None)
        img = bpy.data.images.get(nombre)
        if img is not None:
            bpy.data.images.remove(img)
        return {'FINISHED'}


class AV_OT_tira_sonido(Operator):
    bl_idname = "audioviz.tira_sonido"
    bl_label = "Audio en el secuenciador"
    bl_description = "Mete o saca del Video Sequencer el audio de la fuente activa"
    bl_options = {'REGISTER', 'UNDO'}

    quitar: BoolProperty(default=False)

    def execute(self, contexto):
        escena = contexto.scene
        fuente = fuente_activa(escena)
        if fuente is None:
            self.report({'ERROR'}, "No hay ninguna fuente activa")
            return {'CANCELLED'}
        aud = fuente.audioviz_audio

        if self.quitar:
            quitar_tira_sonido(escena, fuente)
            self.report({'INFO'}, "Audio sacado del secuenciador")
            return {'FINISHED'}

        ruta = bpy.path.abspath(aud.ruta_audio)
        if not ruta or not os.path.isfile(ruta):
            self.report({'ERROR'}, "Esta fuente no sabe de que archivo viene "
                                   "(pasa con las cargadas desde .json)")
            return {'CANCELLED'}
        marcos = fuente.get(CLAVE_FRAMES)
        inicio = int(marcos[0]) if marcos else escena.frame_start
        try:
            poner_tira_sonido(escena, fuente, ruta, inicio)
        except Exception as e:
            self.report({'ERROR'}, f"No he podido: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"'{os.path.basename(ruta)}' en el secuenciador, "
                              f"con el sonido al hacer scrub activado")
        return {'FINISHED'}


class AV_OT_importar(Operator):
    bl_idname = "audioviz.importar"
    bl_label = "Importar audio"
    bl_description = ("Lee el JSON y crea una fuente de audio nueva. Puedes tener varias "
                      "en la misma escena, cada una con su archivo")
    bl_options = {'REGISTER', 'UNDO'}

    reemplazar: BoolProperty(
        name="Reemplazar",
        description="En vez de crear una fuente nueva, cambia el audio de la fuente activa "
                    "conservando su nombre y todo lo que ya dependa de ella",
        default=False,
    )

    def execute(self, contexto):
        escena = contexto.scene
        aj = escena.audioviz

        if not aj.ruta_json:
            self.report({'ERROR'}, "Elige primero el archivo .json")
            return {'CANCELLED'}
        ruta = bpy.path.abspath(aj.ruta_json)
        if not os.path.isfile(ruta):
            self.report({'ERROR'}, f"No existe el archivo: {ruta}")
            return {'CANCELLED'}

        try:
            fotogramas, matriz, n_bandas = leer_json(ruta)
        except Exception as e:
            self.report({'ERROR'}, f"No he podido leer el JSON: {e}")
            return {'CANCELLED'}

        col = obtener_coleccion(escena)
        nombre_corto = os.path.splitext(os.path.basename(ruta))[0]

        if self.reemplazar:
            empty = fuente_activa(escena)
            if empty is None:
                self.report({'ERROR'}, "No hay ninguna fuente activa que reemplazar")
                return {'CANCELLED'}
        else:
            # Una fuente nueva por cada audio. El nombre sale del archivo, que en
            # el outliner se lee muchisimo mejor que 'AudioBands.003'.
            empty = bpy.data.objects.new(f"{PREFIJO_FUENTE}_{nombre_corto}", None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.3
            col.objects.link(empty)

        if empty.name not in col.objects:
            for c in list(empty.users_collection):
                c.objects.unlink(empty)
            col.objects.link(empty)

        fotogramas_blender = [f + aj.fotograma_inicial for f in fotogramas]
        # Un .json solo lleva la mezcla: el estereo hay que sacarlo del audio.
        instalar_fuente(escena, empty, fotogramas_blender, {'MONO': matriz}, n_bandas,
                        nombre_corto, aj.fps)

        if aj.ajustar_escena:
            # Con varias fuentes, la linea de tiempo tiene que abarcarlas todas.
            ajustar_rango_escena(escena, fotogramas_blender, aj.fps)

        n = len(fotogramas)
        empty.audioviz_audio.info = (
            f"{n_bandas} bandas · {n} fotogramas · {fotogramas_blender[0]}-"
            f"{fotogramas_blender[-1]} · {n / max(aj.fps, 1):.1f}s")
        aj.fuente = empty

        verbo = "Reemplazado" if self.reemplazar else "Creada la fuente"
        self.report({'INFO'}, f"{verbo} '{empty.name}': {n_bandas} bandas x {n} fotogramas")
        return {'FINISHED'}


class AV_OT_reaplicar(Operator):
    bl_idname = "audioviz.reaplicar"
    bl_label = "Aplicar suavizado"
    bl_description = ("Recalcula la animacion desde los valores originales del JSON. "
                      "Normalmente no hace falta: se aplica solo al mover los deslizadores")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        empty = fuente_activa(contexto.scene)
        if empty is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}
        aud = empty.audioviz_audio
        if not aplicar_suavizado(empty, aud.ataque, aud.caida, aud.fps):
            self.report({'ERROR'}, "Esta fuente no guarda los valores originales. Vuelve a importar el JSON")
            return {'CANCELLED'}
        self.report({'INFO'}, f"'{empty.name}': ataque {aud.ataque:.2f}s, caida {aud.caida:.2f}s")
        return {'FINISHED'}


class AV_OT_quitar_suavizado(Operator):
    bl_idname = "audioviz.quitar_suavizado"
    bl_label = "Sin suavizado"
    bl_description = "Pone ataque y caida a 0 y devuelve la animacion a los valores originales"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        empty = fuente_activa(contexto.scene)
        if empty is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}
        empty.audioviz_audio.ataque = 0.0
        empty.audioviz_audio.caida = 0.0  # el update() ya recalcula la animacion
        return {'FINISHED'}


def detectar_compas_en(fuente, escena=None):
    """Busca el tempo de una fuente y le deja el compas puesto.

    Devuelve (True, mensaje) si lo ha encontrado, (False, motivo) si no. Lo usan
    el boton y la deteccion automatica al cargar, para que hagan exactamente lo
    mismo y no haya dos caminos que puedan separarse con el tiempo.
    """
    if np is None:
        return False, "hace falta numpy y no esta disponible"
    if fuente is None:
        return False, "no hay ninguna fuente"

    crudos = leer_crudos(fuente)
    if crudos is None:
        return False, "esta fuente no guarda los valores originales"
    _fotogramas, matriz, _n = crudos
    aud = fuente.audioviz_audio

    novedad = funcion_novedad(matriz, aud.fps, aud.banda_min_compas,
                              aud.banda_max_compas)
    if novedad is None:
        return False, "el audio es demasiado plano o demasiado corto"

    resultado = detectar_tempo(novedad, aud.fps)
    if resultado is None:
        return False, "no he encontrado un ritmo claro"

    bpm, desfase, nitidez = resultado

    # La prueba de verdad: partir el tema y ver si los trozos coinciden. Un
    # tempo inventado sobre audio sin ritmo no sobrevive a esto.
    coinciden, trozos = coherencia_tempo(novedad, aud.fps, bpm)
    if trozos and coinciden * 3 < trozos:
        return False, (f"no hay un ritmo estable (solo {coinciden} de {trozos} "
                       f"trozos del tema coinciden). Si sabes el tempo, ponlo "
                       f"a mano en la casilla BPM")

    aud.bpm_detectado = bpm
    aud.compas_acuerdo = coinciden
    aud.compas_trozos = trozos
    aud.fiabilidad_compas = (coinciden / trozos if trozos
                             else 1.0 - 1.0 / max(nitidez, 1.0))
    aud.compas_activo = True
    aud.bpm = bpm                      # su update ya genera las curvas
    aud.desfase_compas = desfase

    generar_compas(fuente, escena)
    respaldo = (f"{coinciden} de {trozos} trozos del tema coinciden" if trozos
                else "tema corto: no he podido comprobarlo por trozos")
    return True, (f"{bpm:.2f} BPM, primer pulso en el fotograma {desfase:.1f} "
                  f"({respaldo})")


class AV_OT_detectar_compas(Operator):
    bl_idname = "audioviz.detectar_compas"
    bl_label = "Detectar compas"
    bl_description = ("Vuelve a buscar el tempo. Se hace solo al cargar un audio; esto "
                      "sirve para repetirlo despues de tocar las bandas que se miran")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        fuente = fuente_activa(contexto.scene)
        if fuente is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}

        bien, mensaje = detectar_compas_en(fuente, contexto.scene)
        if not bien:
            self.report({'ERROR'}, mensaje[0].upper() + mensaje[1:])
            return {'CANCELLED'}
        self.report({'INFO'}, mensaje)
        return {'FINISHED'}


class AV_OT_quitar_compas(Operator):
    bl_idname = "audioviz.quitar_compas"
    bl_label = "Quitar compas"
    bl_description = "Borra las propiedades del compas, su animacion y los marcadores"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        fuente = fuente_activa(contexto.scene)
        if fuente is None:
            return {'CANCELLED'}
        fuente.audioviz_audio.compas_activo = False
        quitar_compas(fuente, contexto.scene)
        self.report({'INFO'}, "Compas eliminado")
        return {'FINISHED'}


class AV_OT_medio_tempo(Operator):
    bl_idname = "audioviz.medio_tempo"
    bl_label = "x0.5 / x2"
    bl_description = ("Parte o dobla el tempo. Los detectores de ritmo confunden a veces "
                      "un tema con el doble o la mitad de su tempo: esto lo arregla de un clic")
    bl_options = {'REGISTER', 'UNDO'}

    factor: FloatProperty(default=2.0)

    def execute(self, contexto):
        fuente = fuente_activa(contexto.scene)
        if fuente is None:
            return {'CANCELLED'}
        aud = fuente.audioviz_audio
        nuevo = aud.bpm * self.factor
        if not (20.0 <= nuevo <= 400.0):
            self.report({'WARNING'}, f"{nuevo:.0f} BPM se sale del rango")
            return {'CANCELLED'}
        aud.bpm = nuevo
        self.report({'INFO'}, f"Ahora {nuevo:.2f} BPM")
        return {'FINISHED'}


class AV_OT_crear_barras(Operator):
    bl_idname = "audioviz.crear_barras"
    bl_label = "Crear barras"
    bl_description = "Crea una fila de barras enganchadas a las bandas mediante drivers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        aj = contexto.scene.audioviz
        empty = fuente_activa(contexto.scene)
        if empty is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}

        bandas = bandas_de(empty)
        if not bandas:
            self.report({'ERROR'}, f"'{empty.name}' no tiene bandas")
            return {'CANCELLED'}

        col = obtener_coleccion(contexto.scene)
        n = len(bandas)
        # El nombre lleva el de la fuente para que crear las barras de un audio
        # no borre las de otro.
        prefijo = f"AV_Barra_{etiqueta_fuente(empty)}_"

        for ob in [o for o in col.objects if o.name.startswith(prefijo)]:
            malla = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if malla and malla.users == 0:
                bpy.data.meshes.remove(malla)

        canal = canal_util(empty, aj.barras_canal)
        columnas = reparto_estereo(bandas, canal)
        # Centramos la fila en el origen.
        x0 = -(len(columnas) - 1) * aj.separacion / 2.0

        for k, (i, canal_col) in enumerate(columnas):
            nombre = f"{prefijo}{k}"
            malla = crear_malla_barra(nombre, aj.ancho)
            ob = bpy.data.objects.new(nombre, malla)
            ob.location = (x0 + k * aj.separacion, 0.0, 0.0)
            ob.parent = empty
            ob.matrix_parent_inverse = empty.matrix_world.inverted()
            col.objects.link(ob)

            # Altura = base + banda * altura_maxima (+ el golpe del compas)
            ob["av_banda"] = i
            ob["av_canal"] = canal_col
            poner_driver_barra(ob, empty, aj)

            if aj.emision:
                mat = crear_material(i, n)
                malla.materials.append(mat)
                # El brillo tambien reacciona: 10% fijo + 90% segun la banda,
                # para que en silencio se intuya el color y no quede negro.
                arbol = mat.node_tree
                if arbol.animation_data is None:
                    arbol.animation_data_create()
                anadir_driver(
                    arbol, 'nodes["Emision"].inputs[1].default_value', -1, empty,
                    {"b": banda_de(empty, canal_col, i)},
                    f"{aj.fuerza_emision * 0.1:.6f} + b * {aj.fuerza_emision * 0.9:.6f}",
                )

        self.report({'INFO'}, f"{n} barras enganchadas a '{empty.name}'")
        return {'FINISHED'}


class AV_OT_crear_led(Operator):
    bl_idname = "audioviz.crear_led"
    bl_label = "Crear ecualizador de LEDs"
    bl_description = "Columnas de cubos sueltos que se encienden de abajo arriba, de verde a rojo"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        aj = contexto.scene.audioviz
        empty = fuente_activa(contexto.scene)
        if empty is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}

        bandas = bandas_de(empty)
        if not bandas:
            self.report({'ERROR'}, f"'{empty.name}' no tiene bandas")
            return {'CANCELLED'}

        col = obtener_coleccion(contexto.scene)
        etiqueta = etiqueta_fuente(empty)
        prefijo = f"{PREFIJO_LED}_{etiqueta}_"
        for ob in [o for o in col.objects if o.name.startswith(prefijo)]:
            bpy.data.objects.remove(ob, do_unlink=True)
        n_seg = aj.led_segmentos
        malla = crear_malla_led(etiqueta, aj.led_ancho, aj.led_alto)
        malla.materials.append(crear_material_led(etiqueta, aj.led_brillo, aj.led_apagado))

        canal = canal_util(empty, aj.led_canal)
        columnas = reparto_estereo(bandas, canal)
        x0 = -(len(columnas) - 1) * aj.led_separacion / 2.0

        for k, (i, canal_col) in enumerate(columnas):
            for j in range(n_seg):
                ob = bpy.data.objects.new(f"{prefijo}{k}_{j}", malla)
                ob.location = (x0 + k * aj.led_separacion, 0.0, (j + 0.5) * aj.led_paso)
                ob.parent = empty
                ob.matrix_parent_inverse = empty.matrix_world.inverted()
                col.objects.link(ob)

                # Altura del segmento dentro de la columna: fija, da el color.
                ob["av_nivel"] = j / max(n_seg - 1, 1)
                ob.id_properties_ui("av_nivel").update(min=0.0, max=1.0)

                # Encendido: driver contra la banda, y el pulso empuja hacia
                # arriba para que en cada golpe se enciendan segmentos de mas.
                ob["av_on"] = 0.0
                ob.id_properties_ui("av_on").update(min=0.0, max=1.0)
                ob["av_banda"] = i
                ob["av_segmento"] = j
                ob["av_canal"] = canal_col
                poner_driver_led(ob, empty, aj, n_seg)

        total = len(columnas) * n_seg
        self.report({'INFO'}, f"{total} segmentos ({len(columnas)} columnas x {n_seg}"
                              f"{', en estereo' if canal == 'ESTEREO' else ''})")
        return {'FINISHED'}


class AV_OT_crear_plexus(Operator):
    bl_idname = "audioviz.crear_plexus"
    bl_label = "Crear plexus"
    bl_description = ("Anade un plexus nuevo e independiente a la escena. Puedes tener "
                      "todos los que quieras, cada uno con su propia configuracion")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        if np is None:
            self.report({'ERROR'}, "Este preset necesita numpy y no esta disponible")
            return {'CANCELLED'}
        empty = fuente_activa(contexto.scene)
        if empty is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}

        col = obtener_coleccion(contexto.scene)
        existentes = plexus_de_la_escena(contexto.scene)

        # Siempre uno nuevo: Blender le pone solo el sufijo .001, .002...
        me = bpy.data.meshes.new(NOMBRE_PLEXUS)
        ob = bpy.data.objects.new(NOMBRE_PLEXUS, me)
        col.objects.link(ob)
        ob.parent = empty
        ob.matrix_parent_inverse = empty.matrix_world.inverted()

        modif = ob.modifiers.new("AV_Plexus", 'NODES')
        modif.node_group = obtener_nodos_plexus()

        p = ob.audioviz_plex
        p.es_plexus = True
        p.fuente = empty          # escucha la fuente activa; cambiable despues
        # Semilla distinta para cada uno: si no, dos plexus recien creados
        # saldrian identicos y superpuestos.
        p.semilla = len(existentes) * 17 % 10000

        actualizar_material_plexus(ob)
        aplicar_estilo_plexus(ob)
        aristas = reconstruir_plexus(contexto.scene, ob)

        # Lo dejamos seleccionado y activo para que el panel pase a editarlo.
        for o in contexto.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        contexto.view_layer.objects.active = ob

        self.report({'INFO'}, f"'{ob.name}' escuchando a '{empty.name}' "
                              f"({p.puntos} puntos, {aristas} lineas). "
                              f"Total de plexus: {len(existentes) + 1}")
        return {'FINISHED'}


class AV_OT_crear_enjambre(Operator):
    bl_idname = "audioviz.crear_enjambre"
    bl_label = "Crear enjambre"
    bl_description = ("Nube de particulas girando alrededor de un centro. Cada banda "
                      "empuja y enciende las suyas, y las ondas del compas la barren")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        if np is None:
            self.report({'ERROR'}, "Este preset necesita numpy y no esta disponible")
            return {'CANCELLED'}
        empty = fuente_activa(contexto.scene)
        if empty is None:
            self.report({'ERROR'}, "Importa primero un audio")
            return {'CANCELLED'}

        col = obtener_coleccion(contexto.scene)
        existentes = enjambres_de_la_escena(contexto.scene)

        me = bpy.data.meshes.new(NOMBRE_ENJAMBRE)
        ob = bpy.data.objects.new(NOMBRE_ENJAMBRE, me)
        col.objects.link(ob)
        ob.parent = empty
        ob.matrix_parent_inverse = empty.matrix_world.inverted()

        modif = ob.modifiers.new("AV_Enjambre", 'NODES')
        modif.node_group = obtener_nodos_enjambre()

        p = ob.audioviz_enj
        p.es_enjambre = True
        p.fuente = empty
        p.semilla = len(existentes) * 23 % 10000
        # Que cubra todas las bandas que tenga el audio, no solo las 8 de por
        # defecto: con 16 bandas la mitad del espectro se quedaria fuera.
        p.banda_max = max(int(empty.get(CLAVE_BANDAS, 8)) - 1, 0)

        actualizar_material_enjambre(ob)
        aplicar_estilo_enjambre(ob)
        n = reconstruir_enjambre(contexto.scene, ob)

        for o in contexto.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        contexto.view_layer.objects.active = ob

        self.report({'INFO'}, f"'{ob.name}' escuchando a '{empty.name}' "
                              f"({n} particulas). Total de enjambres: "
                              f"{len(existentes) + 1}")
        return {'FINISHED'}


class AV_OT_crear_pulso(Operator):
    bl_idname = "audioviz.crear_pulso"
    bl_label = "Crear cubos del compas"
    bl_description = ("Una fila de cubos, uno por tiempo del compas. En cada pulso crece "
                      "y se enciende el que toca, asi se ve el ritmo y en que tiempo va")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        escena = contexto.scene
        aj = escena.audioviz
        fuente = fuente_activa(escena)
        if fuente is None:
            self.report({'ERROR'}, "Carga primero un audio")
            return {'CANCELLED'}
        if not tiene_compas(fuente):
            self.report({'ERROR'}, "Detecta primero el compas (seccion 3 del panel)")
            return {'CANCELLED'}

        col = obtener_coleccion(escena)
        etiqueta = etiqueta_fuente(fuente)
        prefijo = f"{PREFIJO_PULSO_VIS}_{etiqueta}_"
        for ob in [o for o in col.objects if o.name.startswith(prefijo)]:
            bpy.data.objects.remove(ob, do_unlink=True)

        n = max(int(fuente.audioviz_audio.pulsos_por_compas), 1)
        malla = crear_malla_cubo(f"{prefijo}malla", aj.pulso_lado)

        for k in range(n):
            # Cada cubo lleva su propia copia de la malla porque tambien lleva
            # su propio material, y el material va pegado a la malla.
            ob = bpy.data.objects.new(f"{prefijo}{k}", malla.copy())
            ob.location = (0.0, 0.0, 0.0)
            ob.parent = fuente
            ob.matrix_parent_inverse = fuente.matrix_world.inverted()
            col.objects.link(ob)
            ob["av_pulso_indice"] = k

            ob.data.materials.append(
                crear_material_pulso(k, n, aj.pulso_brillo, aj.pulso_apagado))
            configurar_cubo_pulso(ob, fuente, aj, k, n)

        bpy.data.meshes.remove(malla)     # solo era la plantilla

        self.report({'INFO'}, f"{n} cubos de compas para '{fuente.name}' "
                              f"({fuente.audioviz_audio.bpm:.1f} BPM)")
        return {'FINISHED'}


class AV_OT_crear_paisaje(Operator):
    bl_idname = "audioviz.crear_paisaje"
    bl_label = "Crear paisaje"
    bl_description = ("Rejilla de montanas que avanza: a lo ancho las frecuencias, a lo "
                      "largo el tiempo. Cada fila es un instante del pasado")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        if np is None:
            self.report({'ERROR'}, "Hace falta numpy y no esta disponible")
            return {'CANCELLED'}
        escena = contexto.scene
        fuente = fuente_activa(escena)
        if fuente is None:
            self.report({'ERROR'}, "Carga primero un audio")
            return {'CANCELLED'}

        col = obtener_coleccion(escena)
        me = bpy.data.meshes.new(NOMBRE_PAISAJE)
        ob = bpy.data.objects.new(NOMBRE_PAISAJE, me)
        col.objects.link(ob)
        ob.parent = fuente
        ob.matrix_parent_inverse = fuente.matrix_world.inverted()

        p = ob.audioviz_paisaje
        p.es_paisaje = True
        p.fuente = fuente
        n_bandas = int(fuente.get(CLAVE_BANDAS, 8))
        p.banda_max = max(0, n_bandas - 1)

        actualizar_material_paisaje(ob)
        aplicar_estilo_paisaje(ob)
        celdas = actualizar_paisaje(escena, ob)

        for o in contexto.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        contexto.view_layer.objects.active = ob

        segundos = p.filas * p.fotogramas_por_fila / max(fuente.audioviz_audio.fps, 1)
        self.report({'INFO'}, f"'{ob.name}' escuchando a '{fuente.name}': "
                              f"{p.filas}x{p.columnas} ({celdas} celdas), "
                              f"{segundos:.1f} s de historia a la vista")
        return {'FINISHED'}


class AV_OT_hornear(Operator):
    bl_idname = "audioviz.hornear"
    bl_label = "Hornear"
    bl_description = ("Deja una copia con la animacion metida en claves de forma, que "
                      "funciona sin la extension: para mandar el .blend a una granja de "
                      "render o a alguien que no la tenga. El original no se toca")
    bl_options = {'REGISTER', 'UNDO'}

    todo_el_rango: BoolProperty(
        name="Todo el rango de la escena",
        description="Desmarcado, hornea solo desde el fotograma actual hasta el final",
        default=True,
    )

    def execute(self, contexto):
        escena = contexto.scene
        ob = contexto.object
        if not (es_plexus(ob) or es_paisaje(ob) or es_enjambre(ob)):
            self.report({'ERROR'}, "Selecciona un plexus, un paisaje o un enjambre")
            return {'CANCELLED'}

        inicio = escena.frame_start if self.todo_el_rango else escena.frame_current
        fin = escena.frame_end
        if fin < inicio:
            self.report({'ERROR'}, "El rango de la escena esta al reves")
            return {'CANCELLED'}
        if fin - inicio > 4000:
            self.report({'ERROR'}, f"{fin - inicio + 1} fotogramas es demasiado; "
                                   "acorta el rango de la escena")
            return {'CANCELLED'}

        objetivos = [ob]
        if es_plexus(ob):
            caras = objeto_caras_de(ob)
            if caras is not None:
                objetivos.append(caras)

        aviso = ""
        if es_plexus(ob):
            estable = estabilidad_aristas(escena, ob)
            if estable < 90.0:
                aviso = (f" · ojo: las conexiones cambian mucho ({estable:.0f}% "
                         f"coinciden), el horneado quedara distinto")

        wm = contexto.window_manager
        wm.progress_begin(0, 100)
        horneados = []
        try:
            for indice, objetivo in enumerate(objetivos):
                def avance(hechos, total, _i=indice, _n=len(objetivos)):
                    wm.progress_update(int((_i + hechos / max(total, 1)) / _n * 100))
                try:
                    horneados.append(hornear_objeto(escena, objetivo, inicio, fin, avance))
                except Exception as e:
                    self.report({'ERROR'}, f"No he podido hornear {objetivo.name}: {e}")
                    return {'CANCELLED'}
        finally:
            wm.progress_end()

        # El horneado se queda visible y el original se aparta, para que no se
        # vean los dos encima.
        for original in objetivos:
            original.hide_viewport = True
            original.hide_render = True
        for h in horneados:
            h.hide_viewport = False
            h.hide_render = False

        for o in contexto.selected_objects:
            o.select_set(False)
        horneados[0].select_set(True)
        contexto.view_layer.objects.active = horneados[0]

        n_claves = fin - inicio + 1
        peso = sum(len(h.data.vertices) for h in horneados) * n_claves * 12 / 1024 / 1024
        self.report({'INFO'},
                    f"{', '.join(h.name for h in horneados)}: {n_claves} fotogramas "
                    f"({inicio}-{fin}), ~{peso:.1f} MB. El original queda oculto{aviso}")
        return {'FINISHED'}


class AV_OT_regenerar_puntos(Operator):
    bl_idname = "audioviz.regenerar_puntos"
    bl_label = "Regenerar puntos"
    bl_description = ("Vuelve a repartir los puntos. Necesario si has movido, escalado o "
                      "editado el modelo del que salen: el plexus nace encima de el pero "
                      "despues no lo persigue")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        ob = contexto.object
        if not es_plexus(ob):
            self.report({'ERROR'}, "Selecciona un plexus")
            return {'CANCELLED'}
        p0, _ = disposicion(ob, forzar=True)
        reconstruir_plexus(contexto.scene, ob)
        self.report({'INFO'}, f"{len(p0)} puntos repartidos de nuevo")
        return {'FINISHED'}


class AV_OT_ajustar_distancia(Operator):
    bl_idname = "audioviz.ajustar_distancia"
    bl_label = "Ajustar al tamano"
    bl_description = ("Pone la distancia de union a partir de lo separados que estan los "
                      "puntos, para que salga una malla razonable sin ir a tientas")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        ob = contexto.object
        if not es_plexus(ob):
            self.report({'ERROR'}, "Selecciona un plexus")
            return {'CANCELLED'}
        sugerida = distancia_sugerida(ob)
        if sugerida <= 0.0:
            self.report({'ERROR'}, "No he podido calcularla")
            return {'CANCELLED'}
        ob.audioviz_plex.distancia = sugerida
        self.report({'INFO'}, f"Distancia de union: {sugerida:.3f}")
        return {'FINISHED'}


class AV_OT_seleccionar_plexus(Operator):
    bl_idname = "audioviz.seleccionar_plexus"
    bl_label = "Seleccionar plexus"
    bl_description = "Hace activo este plexus para poder editar sus ajustes en el panel"
    bl_options = {'REGISTER', 'UNDO'}

    nombre: StringProperty()

    def execute(self, contexto):
        ob = bpy.data.objects.get(self.nombre)
        if ob is None:
            self.report({'ERROR'}, f"Ya no existe '{self.nombre}'")
            return {'CANCELLED'}
        for o in contexto.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        contexto.view_layer.objects.active = ob
        return {'FINISHED'}


class AV_OT_material_unico(Operator):
    bl_idname = "audioviz.material_unico"
    bl_label = "Desvincular material"
    bl_description = ("Da a este plexus una copia propia del material. Util despues de "
                      "duplicar con Shift+D, que deja el material compartido")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, contexto):
        ob = contexto.object
        if not es_plexus(ob):
            self.report({'ERROR'}, "Selecciona un plexus")
            return {'CANCELLED'}
        modif = modificador_plexus(ob)
        ident = identificador_entrada(modif.node_group, "Material")
        antiguo = modif[ident]
        modif[ident] = antiguo.copy() if antiguo is not None else None
        actualizar_material_plexus(ob)
        self.report({'INFO'}, f"'{ob.name}' ya tiene material propio")
        return {'FINISHED'}


class AV_OT_limpiar(Operator):
    bl_idname = "audioviz.limpiar"
    bl_label = "Borrar todo"
    bl_description = "Elimina el Empty, las barras, los materiales y la animacion generados"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, contexto, evento):
        return contexto.window_manager.invoke_confirm(self, evento)

    def execute(self, contexto):
        # Primero las tiras de sonido: despues de borrar los Empty ya no
        # sabriamos cuales eran nuestras.
        for fuente in fuentes_de_la_escena(contexto.scene):
            quitar_tira_sonido(contexto.scene, fuente)
        tiras = tiras_de(contexto.scene)
        if tiras is not None:
            for t in [x for x in tiras if x.name.startswith(PREFIJO_TIRA)]:
                sonido = getattr(t, "sound", None)
                tiras.remove(t)
                if sonido is not None and sonido.users == 0:
                    bpy.data.sounds.remove(sonido)

        col = bpy.data.collections.get(NOMBRE_COLECCION)
        if col is not None:
            for ob in list(col.objects):
                malla = ob.data if ob.type == 'MESH' else None
                if ob.animation_data and ob.animation_data.action:
                    accion = ob.animation_data.action
                    ob.animation_data.action = None
                    if accion.users == 0:
                        bpy.data.actions.remove(accion)
                bpy.data.objects.remove(ob, do_unlink=True)
                if malla and malla.users == 0:
                    bpy.data.meshes.remove(malla)
            bpy.data.collections.remove(col)

        _espectros.clear()
        for img in [i for i in bpy.data.images if i.name.startswith(NOMBRE_IMAGEN)]:
            bpy.data.images.remove(img)

        for mat in list(bpy.data.materials):
            if mat.users == 0 and mat.name.startswith(
                    (PREFIJO_MATERIAL, NOMBRE_MAT_LED, NOMBRE_MAT_PLEXUS,
                     NOMBRE_MAT_CARAS, NOMBRE_MAT_PAISAJE, NOMBRE_MAT_PULSO,
                     NOMBRE_MAT_ENJAMBRE)):
                bpy.data.materials.remove(mat)

        for me in list(bpy.data.meshes):
            if me.users == 0 and me.name.startswith(PREFIJO_LED):
                bpy.data.meshes.remove(me)

        for nombre in (NOMBRE_GN_PLEXUS, NOMBRE_GN_ENJAMBRE):
            ng = bpy.data.node_groups.get(nombre)
            if ng is not None and ng.users == 0:
                bpy.data.node_groups.remove(ng)

        contexto.scene.audioviz.info = ""
        self.report({'INFO'}, "Limpiado")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# PANEL
# ---------------------------------------------------------------------------

class AV_PT_panel(Panel):
    bl_label = "Audio Viz"
    bl_idname = "AV_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Viz"

    def draw(self, contexto):
        d = self.layout
        aj = contexto.scene.audioviz

        lista = fuentes_de_la_escena(contexto.scene)
        empty = fuente_activa(contexto.scene)

        # --- 1. cargar audios ---
        caja = d.box()
        caja.label(text=f"1. Audios cargados ({len(lista)})", icon='SOUND')
        if lista:
            caja.prop(aj, "fuente", text="Activa")

        fila = caja.row(align=True)
        fila.prop(aj, "fotograma_inicial")
        fila.prop(aj, "fps")
        caja.prop(aj, "ajustar_escena")
        caja.prop(aj, "detectar_compas_al_cargar")

        # Analizar aqui mismo: no hace falta ningun programa de fuera.
        col = caja.column(align=True)
        col.scale_y = 1.3
        col.operator("audioviz.analizar_audio", text="Analizar un audio...",
                     icon='FILE_SOUND').reemplazar = False
        if empty is not None:
            fila = caja.row(align=True)
            fila.operator("audioviz.analizar_audio", text="Reemplazar el audio de esta fuente",
                          icon='FILE_REFRESH').reemplazar = True

        cabecera = caja.row(align=True)
        cabecera.prop(aj, "mostrar_avanzado", emboss=False,
                      icon='TRIA_DOWN' if aj.mostrar_avanzado else 'TRIA_RIGHT',
                      text="Ajustes del analisis")
        if aj.mostrar_avanzado:
            sub = caja.column(align=True)
            sub.prop(aj, "bandas")
            sub.prop(aj, "ventana")
            sub.prop(aj, "fmin")
            sub.prop(aj, "fmax")
            sub.prop(aj, "rango_db")
            sub.prop(aj, "norm")
            sub.prop(aj, "analizar_estereo")
            sub.prop(aj, "guardar_json")
            sub.label(text="El suavizado se ajusta luego, en vivo", icon='INFO')

        cabecera = caja.row(align=True)
        cabecera.label(text="o cargar un .json ya analizado:", icon='FILE_BLANK')
        caja.prop(aj, "ruta_json", text="")
        fila = caja.row(align=True)
        fila.operator("audioviz.importar", text="Cargar JSON",
                      icon='IMPORT').reemplazar = False
        if empty is not None:
            fila.operator("audioviz.importar", text="Reemplazar con JSON",
                          icon='FILE_REFRESH').reemplazar = True

        if empty is not None and empty.audioviz_audio.info:
            caja.label(text=empty.audioviz_audio.info, icon='CHECKMARK')

        if empty is not None:
            aud_act = empty.audioviz_audio
            tira = tira_de_fuente(contexto.scene, empty)
            fila = caja.row(align=True)
            if tira is not None:
                fila.prop(aud_act, "oir_audio",
                          icon='SEQ_SEQUENCER' if aud_act.oir_audio else 'MUTE_IPO_ON')
                fila.operator("audioviz.tira_sonido", text="", icon='X').quitar = True
            elif aud_act.ruta_audio:
                fila.operator("audioviz.tira_sonido", text="Recuperar el audio en la pista",
                              icon='SEQ_SEQUENCER').quitar = False
            else:
                fila.label(text="sin audio: viene de un .json", icon='INFO')

        if empty is None:
            d.separator()
            d.operator("audioviz.limpiar", icon='TRASH')
            return

        aud = empty.audioviz_audio
        hay_crudos = leer_crudos(empty) is not None

        # --- 2. suavizado, propio de cada audio ---
        caja = d.box()
        caja.label(text=f"2. Suavizado de '{empty.name}'", icon='SMOOTHCURVE')
        sub = caja.column()
        sub.enabled = hay_crudos
        sub.prop(aud, "fps")
        sub.prop(aud, "ataque", slider=True)
        sub.prop(aud, "caida", slider=True)

        # Traducimos los segundos a fotogramas, que es lo que se ve en la
        # linea de tiempo, para que el ajuste sea comprobable de un vistazo.
        if aud.caida > 0.0:
            sub.label(text=f"un pico cae a cero en {aud.caida * aud.fps:.0f} fotogramas",
                      icon='IPO_LINEAR')
        else:
            sub.label(text="sigue al audio tal cual", icon='IPO_CONSTANT')

        fila = sub.row(align=True)
        fila.operator("audioviz.quitar_suavizado", icon='X')
        fila.operator("audioviz.reaplicar", icon='FILE_REFRESH')
        if not hay_crudos:
            caja.label(text="Reimporta el JSON para activarlo", icon='INFO')

        # --- 3. compas ---
        caja = d.box()
        caja.label(text="3. Compas y ritmo", icon='PLAY_SOUND')
        sub = caja.column()
        sub.enabled = hay_crudos

        if not aud.compas_activo:
            sub.label(text="sin compas: no lo encontro, o lo quitaste", icon='INFO')
            fila = sub.row(align=True)
            fila.prop(aud, "banda_min_compas", text="Bandas")
            fila.prop(aud, "banda_max_compas", text="a")
            sub.label(text="prueba con otras bandas y vuelve a buscar")
            sub.operator("audioviz.detectar_compas", text="Buscar el compas",
                         icon='TIME')
        else:
            if aud.bpm_detectado > 0.0:
                sub.label(text=f"detectado {aud.bpm_detectado:.2f} BPM",
                          icon='CHECKMARK')
                if aud.compas_trozos:
                    sub.label(text=f"    {aud.compas_acuerdo} de "
                                   f"{aud.compas_trozos} trozos del tema coinciden",
                              icon='DOT')
                else:
                    sub.label(text="    tema corto: no he podido comprobarlo "
                                   "por trozos", icon='DOT')
            fila = sub.row(align=True)
            fila.prop(aud, "bpm")
            fila.operator("audioviz.medio_tempo", text="x2").factor = 2.0
            fila.operator("audioviz.medio_tempo", text="/2").factor = 0.5
            sub.prop(aud, "desfase_compas")
            sub.prop(aud, "pulsos_por_compas")
            sub.prop(aud, "caida_pulso")
            sub.prop(aud, "marcadores")

            periodo = aud.fps * 60.0 / max(aud.bpm, 1e-6)
            sub.label(text=f"un pulso cada {periodo:.2f} fotogramas "
                           f"({60.0 / max(aud.bpm, 1e-6):.3f} s)", icon='IPO_LINEAR')
            fila = sub.row(align=True)
            fila.operator("audioviz.detectar_compas", text="Volver a detectar",
                          icon='FILE_REFRESH')
            fila.operator("audioviz.quitar_compas", icon='X')

        # --- ver el analisis ---
        caja = d.box()
        caja.label(text="Ver el analisis", icon='SEQ_HISTOGRAM')
        nombre_img = f"{NOMBRE_IMAGEN}_{etiqueta_fuente(empty)}"
        img = bpy.data.images.get(nombre_img)
        if es_estereo(empty):
            caja.prop(aj, "canal_analisis", text="")
        fila = caja.row(align=True)
        fila.operator("audioviz.ver_analisis",
                      text="Actualizar" if img else "Dibujar el espectrograma",
                      icon='SEQ_HISTOGRAM')
        if img is not None:
            fila.operator("audioviz.quitar_analisis", text="", icon='X')
            # La vista previa dentro del panel; si Blender no le ha dado icono,
            # al menos queda dicho como verla.
            icono = getattr(img.preview, "icon_id", 0) if img.preview else 0
            if icono:
                caja.template_icon(icon_value=icono, scale=7.0)
            caja.label(text=f"'{img.name}' · {img.size[0]}x{img.size[1]}",
                       icon='IMAGE_DATA')
            caja.label(text="abrelo en un editor de imagenes para verlo grande")

        caja = d.box()
        caja.label(text="Valores en este fotograma", icon='GRAPH')
        canales_ver = ('MONO', 'IZQ', 'DER') if es_estereo(empty) else ('MONO',)
        for canal_ver in canales_ver:
            prefijo_ver = PREFIJOS_CANAL[canal_ver]
            if canal_ver != 'MONO':
                caja.separator()
                caja.label(text={'IZQ': "canal izquierdo",
                                 'DER': "canal derecho"}[canal_ver])
            for i in bandas_de(empty, canal_ver):
                caja.prop(empty, f'["{prefijo_ver}{i}"]', text=f"{prefijo_ver}{i}")
        if aud.compas_activo:
            caja.separator()
            for clave in CLAVES_COMPAS:
                if clave in empty:
                    caja.prop(empty, f'["{clave}"]', text=clave)

        d.separator()
        d.operator("audioviz.limpiar", icon='TRASH')


class AV_PT_base_preset(Panel):
    """Base comun de los subpaneles de presets."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Viz"
    bl_parent_id = "AV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}


def aviso_compas(disposicion, fuente):
    """Recuerda que hay que detectar el compas antes de que el control sirva."""
    if not tiene_compas(fuente):
        disposicion.label(text="detecta el compas arriba para usarlo", icon='INFO')
        return False
    return True


def selector_canal(disposicion, datos, propiedad, fuente):
    """El canal solo se ofrece si la fuente guarda los dos por separado."""
    if es_estereo(fuente):
        disposicion.prop(datos, propiedad)
        return True
    disposicion.label(text="mono: marca 'Analizar en estereo' al cargar", icon='INFO')
    return False


class AV_PT_barras(AV_PT_base_preset, Panel):
    bl_label = "Preset: barras"
    bl_idname = "AV_PT_barras"

    def draw(self, contexto):
        d = self.layout
        aj = contexto.scene.audioviz
        selector_canal(d, aj, "barras_canal", fuente_activa(contexto.scene))
        d.prop(aj, "altura")
        d.prop(aj, "base")
        d.prop(aj, "ancho")
        d.prop(aj, "separacion")
        d.prop(aj, "emision")
        sub = d.row()
        sub.enabled = aj.emision
        sub.prop(aj, "fuerza_emision")

        col = d.column(align=True)
        col.prop(aj, "barras_pulso")
        aviso_compas(col, fuente_activa(contexto.scene))

        d.operator("audioviz.crear_barras", icon='PLUS')


class AV_PT_pulso(AV_PT_base_preset, Panel):
    bl_label = "Preset: el compas en cubos"
    bl_idname = "AV_PT_pulso"

    def draw(self, contexto):
        d = self.layout
        aj = contexto.scene.audioviz
        fuente = fuente_activa(contexto.scene)

        col = d.column(align=True)
        col.prop(aj, "pulso_lado")
        col.prop(aj, "pulso_separacion")
        col.prop(aj, "pulso_crecimiento")
        col = d.column(align=True)
        col.prop(aj, "pulso_brillo")
        col.prop(aj, "pulso_apagado")

        if tiene_compas(fuente):
            n = fuente.audioviz_audio.pulsos_por_compas
            d.label(text=f"{n} cubos, uno por tiempo · "
                         f"{fuente.audioviz_audio.bpm:.1f} BPM", icon='PLAY_SOUND')
        else:
            d.label(text="detecta el compas arriba para usarlo", icon='INFO')
        sub = d.column()
        sub.enabled = tiene_compas(fuente)
        sub.operator("audioviz.crear_pulso", icon='PLUS')


class AV_PT_led(AV_PT_base_preset, Panel):
    bl_label = "Preset: ecualizador LED"
    bl_idname = "AV_PT_led"

    def draw(self, contexto):
        d = self.layout
        aj = contexto.scene.audioviz

        selector_canal(d, aj, "led_canal", fuente_activa(contexto.scene))
        col = d.column(align=True)
        col.prop(aj, "led_segmentos")
        col.prop(aj, "led_dureza")

        col = d.column(align=True)
        col.prop(aj, "led_ancho")
        col.prop(aj, "led_alto")
        col.prop(aj, "led_paso")
        col.prop(aj, "led_separacion")
        if aj.led_paso <= aj.led_alto:
            d.label(text="Sube el paso por encima del alto para que haya hueco", icon='INFO')

        col = d.column(align=True)
        col.prop(aj, "led_brillo")
        col.prop(aj, "led_apagado")

        col = d.column(align=True)
        col.prop(aj, "led_pulso", slider=True)
        aviso_compas(col, fuente_activa(contexto.scene))

        empty = bpy.data.objects.get(NOMBRE_EMPTY)
        if empty is not None:
            n_b = len(bandas_de(empty))
            if canal_util(empty, aj.led_canal) == 'ESTEREO':
                n_b *= 2
            d.label(text=f"{n_b * aj.led_segmentos} cubos, 1 malla, 1 material",
                    icon='OUTLINER_OB_MESH')
        d.operator("audioviz.crear_led", icon='PLUS')


class AV_PT_plexus(AV_PT_base_preset, Panel):
    bl_label = "Preset: plexus"
    bl_idname = "AV_PT_plexus"

    def draw(self, contexto):
        d = self.layout

        if np is None:
            d.label(text="Necesita numpy y no esta disponible", icon='ERROR')
            return

        d.operator("audioviz.crear_plexus", text="Anadir un plexus", icon='PLUS')

        lista = plexus_de_la_escena(contexto.scene)
        ob = contexto.object

        if not es_plexus(ob):
            if lista:
                caja = d.box()
                caja.label(text="Elige cual editar:", icon='INFO')
                col = caja.column(align=True)
                for o in lista:
                    col.operator("audioviz.seleccionar_plexus", text=o.name,
                                 icon='OUTLINER_OB_MESH').nombre = o.name
            return

        p = ob.audioviz_plex
        caja = d.box()
        fila = caja.row()
        fila.label(text=ob.name, icon='OUTLINER_OB_MESH')
        fila.label(text=f"{len(lista)} en la escena")

        col = caja.column(align=True)
        col.prop(p, "fuente", text="Audio")
        if not es_fuente(p.fuente):
            col.label(text="Sin audio propio: usa el activo", icon='INFO')

        col = caja.column(align=True)
        col.prop(p, "forma")
        de_modelo = p.forma in FORMAS_DE_MODELO
        if de_modelo:
            col.prop(p, "objeto_origen")
            if p.objeto_origen is None:
                col.label(text="Elige un modelo (mientras, esfera)", icon='ERROR')
            elif p.forma == 'VOLUMEN':
                col.label(text="La malla debe estar cerrada", icon='INFO')
        col.prop(p, "puntos")
        if not de_modelo:
            col.prop(p, "radio")
        col.prop(p, "semilla")
        if de_modelo and p.objeto_origen is not None:
            col.operator("audioviz.regenerar_puntos", icon='FILE_REFRESH')

        col = caja.column(align=True)
        col.label(text="Que frecuencias lo mueven:")
        selector_canal(col, p, "canal",
                       p.fuente if es_fuente(p.fuente) else fuente_activa(contexto.scene))
        col.prop(p, "asignacion")
        fila = col.row(align=True)
        fila.prop(p, "banda_min", text="De")
        fila.prop(p, "banda_max", text="a")
        col.prop(p, "suave")
        col.prop(p, "amplitud")

        col = caja.column(align=True)
        col.label(text="Ritmo:")
        col.prop(p, "pulso_amplitud")

        col = caja.column(align=True)
        col.prop(p, "pulso_onda")
        sub = col.row()
        sub.enabled = p.pulso_onda > 0.0
        sub.prop(p, "onda_grosor", slider=True)

        col = caja.column(align=True)
        col.prop(p, "compas_onda")
        sub = col.row()
        sub.enabled = p.compas_onda > 0.0
        sub.prop(p, "compas_onda_grosor", slider=True)

        fuente_plex = p.fuente if es_fuente(p.fuente) else fuente_activa(contexto.scene)
        if tiene_compas(fuente_plex) and (p.pulso_onda > 0.0 or p.compas_onda > 0.0):
            aud = fuente_plex.audioviz_audio
            periodo = aud.fps * 60.0 / max(aud.bpm, 1e-6)
            partes = []
            if p.pulso_onda > 0.0:
                partes.append(f"pulso: cruza en {periodo:.1f} fotogramas")
            if p.compas_onda > 0.0:
                partes.append(f"compas: en {periodo * aud.pulsos_por_compas:.1f}")
            caja.label(text=" · ".join(partes), icon='IPO_LINEAR')
        aviso_compas(caja, fuente_plex)

        col = caja.column(align=True)
        col.label(text="Como se traza:")
        fila = col.row(align=True)
        fila.prop(p, "distancia")
        fila.operator("audioviz.ajustar_distancia", text="", icon='DRIVER_DISTANCE')
        col.prop(p, "conexiones")
        col.prop(p, "grosor")
        col.prop(p, "tam_punto")

        col = caja.column(align=True)
        col.label(text="Color de las lineas:")
        fila = col.row(align=True)
        fila.prop(p, "color_grave", text="")
        fila.prop(p, "color_medio", text="")
        fila.prop(p, "color_agudo", text="")
        col.prop(p, "brillo")

        col = caja.column(align=True)
        col.prop(p, "caras")
        if p.caras:
            sub = col.column(align=True)
            sub.prop(p, "ratio_caras", slider=True)
            sub.prop(p, "opacidad_caras", slider=True)
            sub.prop(p, "brillo_caras")

            sub = col.column(align=True)
            sub.prop(p, "degradado_caras")
            if p.degradado_caras:
                sub.prop(p, "atributo_caras")
                fila = sub.row(align=True)
                fila.prop(p, "color_caras", text="")
                fila.prop(p, "color_caras_alta", text="")
            else:
                sub.prop(p, "color_caras", text="")

            obc = objeto_caras_de(ob)
            if obc is not None:
                col.label(text=f"'{obc.name}': {len(obc.data.polygons)} caras",
                          icon='MESH_DATA')
            # Sin triangulos cerrados no hay caras que rellenar, y con pocas
            # conexiones por punto casi ninguno se cierra.
            if p.conexiones < 6:
                col.label(text="Sube 'Conexiones por punto' a 6-10 para que", icon='INFO')
                col.label(text="se cierren triangulos y haya membrana")

        mat = material_de_plexus(ob)
        if mat is not None and mat.users > 1:
            aviso = caja.column(align=True)
            aviso.label(text=f"Material compartido con {mat.users - 1} mas", icon='LINKED')
            aviso.operator("audioviz.material_unico", icon='UNLINKED')

        caja.separator()
        caja.operator("audioviz.hornear", icon='FILE_TICK')

        caja.label(text=f"{len(ob.data.vertices)} puntos · {len(ob.data.edges)} lineas ahora",
                   icon='MOD_SIMPLIFY')


class AV_PT_paisaje(AV_PT_base_preset, Panel):
    bl_label = "Preset: paisaje que avanza"
    bl_idname = "AV_PT_paisaje"

    def draw(self, contexto):
        d = self.layout
        if np is None:
            d.label(text="Necesita numpy y no esta disponible", icon='ERROR')
            return

        d.operator("audioviz.crear_paisaje", text="Anadir un paisaje", icon='PLUS')

        lista = paisajes_de_la_escena(contexto.scene)
        ob = contexto.object
        if not es_paisaje(ob):
            if lista:
                caja = d.box()
                caja.label(text="Elige cual editar:", icon='INFO')
                col = caja.column(align=True)
                for o in lista:
                    col.operator("audioviz.seleccionar_plexus", text=o.name,
                                 icon='MESH_GRID').nombre = o.name
            return

        p = ob.audioviz_paisaje
        caja = d.box()
        fila = caja.row()
        fila.label(text=ob.name, icon='MESH_GRID')
        fila.label(text=f"{len(lista)} en la escena")

        col = caja.column(align=True)
        col.prop(p, "fuente", text="Audio")

        col = caja.column(align=True)
        col.label(text="Movimiento:")
        col.prop(p, "direccion")
        col.prop(p, "fotogramas_por_fila")
        fuente = p.fuente if es_fuente(p.fuente) else fuente_activa(contexto.scene)
        fps = fuente.audioviz_audio.fps if fuente is not None else 24
        segundos = p.filas * p.fotogramas_por_fila / max(fps, 1)
        col.label(text=f"{segundos:.1f} s de historia · "
                       f"{1.0 / max(p.fotogramas_por_fila, 1e-6):.2f} filas por fotograma",
                  icon='TIME')

        col = caja.column(align=True)
        col.label(text="Rejilla:")
        col.prop(p, "filas")
        col.prop(p, "columnas")
        col.prop(p, "ancho")
        col.prop(p, "largo")
        col.prop(p, "altura")

        col = caja.column(align=True)
        col.label(text="Repetir a los lados:")
        col.prop(p, "repeticiones")
        sub = col.row()
        sub.enabled = p.repeticiones > 1
        sub.prop(p, "espejo")
        total = columnas_totales(p)
        col.label(text=f"{p.filas} x {total} = {p.filas * total} vertices · "
                       f"{p.ancho * p.repeticiones:.1f} de ancho total", icon='MESH_DATA')

        col = caja.column(align=True)
        col.label(text="Frecuencias:")
        selector_canal(col, p, "canal", fuente)
        fila = col.row(align=True)
        fila.prop(p, "banda_min", text="De")
        fila.prop(p, "banda_max", text="a")
        col.prop(p, "suave")

        col = caja.column(align=True)
        col.label(text="Moldear el relieve:")
        col.prop(p, "ganancia")
        col.prop(p, "curva")
        col.prop(p, "inclinacion", slider=True)
        col.prop(p, "suelo", slider=True)
        col.prop(p, "pulso_altura", slider=True)
        sub = col.row()
        sub.enabled = p.pulso_altura > 0.0
        sub.prop(p, "pulso_extension", slider=True)

        col = caja.column(align=True)
        col.prop(p, "compas_marca", slider=True)
        sub = col.column(align=True)
        sub.enabled = p.compas_marca > 0.0
        sub.prop(p, "compas_marca_lado")
        sub.prop(p, "compas_marca_ancho", slider=True)
        if tiene_compas(fuente) and p.compas_marca > 0.0:
            aud = fuente.audioviz_audio
            filas_compas = (aud.fps * 60.0 / max(aud.bpm, 1e-6)
                            * aud.pulsos_por_compas / max(p.fotogramas_por_fila, 1e-6))
            col.label(text=f"una muesca cada {filas_compas:.0f} filas", icon='IPO_LINEAR')
        aviso_compas(col, fuente)

        col = caja.column(align=True)
        col.label(text="Aspecto:")
        col.prop(p, "modo")
        sub = col.column(align=True)
        sub.enabled = p.modo != 'SOLIDO'
        sub.prop(p, "grosor_malla")
        fila_op = sub.row()
        fila_op.enabled = p.modo == 'AMBOS'
        fila_op.prop(p, "opacidad_superficie", slider=True)
        fila = col.row(align=True)
        fila.prop(p, "color_bajo", text="")
        fila.prop(p, "color_alto", text="")
        col.prop(p, "brillo")
        col.prop(p, "desvanecer")
        sub = col.row()
        sub.enabled = p.desvanecer
        sub.prop(p, "desvanecido", slider=True)

        caja.separator()
        caja.operator("audioviz.hornear", icon='FILE_TICK')


class AV_PT_enjambre(AV_PT_base_preset, Panel):
    bl_label = "Preset: enjambre orbital"
    bl_idname = "AV_PT_enjambre"

    def draw(self, contexto):
        d = self.layout

        if np is None:
            d.label(text="Necesita numpy y no esta disponible", icon='ERROR')
            return

        d.operator("audioviz.crear_enjambre", text="Anadir un enjambre", icon='PLUS')

        lista = enjambres_de_la_escena(contexto.scene)
        ob = contexto.object

        if not es_enjambre(ob):
            if lista:
                caja = d.box()
                caja.label(text="Elige cual editar:", icon='INFO')
                col = caja.column(align=True)
                for o in lista:
                    col.operator("audioviz.seleccionar_plexus", text=o.name,
                                 icon='OUTLINER_OB_MESH').nombre = o.name
            return

        p = ob.audioviz_enj
        fuente = p.fuente if es_fuente(p.fuente) else fuente_activa(contexto.scene)

        caja = d.box()
        fila = caja.row()
        fila.label(text=ob.name, icon='OUTLINER_OB_MESH')
        fila.label(text=f"{len(lista)} en la escena")

        col = caja.column(align=True)
        col.prop(p, "fuente", text="Audio")
        if not es_fuente(p.fuente):
            col.label(text="Sin audio propio: usa el activo", icon='INFO')

        col = caja.column(align=True)
        col.label(text="La nube:")
        col.prop(p, "forma")
        col.prop(p, "particulas")
        col.prop(p, "radio")
        if p.forma != 'ESFERA':
            col.prop(p, "grosor")
        col.prop(p, "semilla")

        col = caja.column(align=True)
        col.label(text="Giro:")
        col.prop(p, "giro")
        col.prop(p, "diferencial")

        col = caja.column(align=True)
        col.label(text="Que frecuencias la mueven:")
        selector_canal(col, p, "canal", fuente)
        col.prop(p, "reparto")
        fila = col.row(align=True)
        fila.prop(p, "banda_min", text="De")
        fila.prop(p, "banda_max", text="a")
        col.prop(p, "suave")

        col = caja.column(align=True)
        col.label(text="Cuanto manda cada zona del espectro:")
        fila = col.row(align=True)
        fila.prop(p, "peso_graves")
        fila.prop(p, "peso_medios")
        fila.prop(p, "peso_agudos")
        if p.peso_graves == p.peso_medios == p.peso_agudos == 1.0:
            col.label(text="los tres a 1: sin tocar", icon='DOT')
        col.label(text="afecta al movimiento y al brillo", icon='INFO')

        col = caja.column(align=True)
        col.label(text="Como reacciona a las frecuencias:")
        col.prop(p, "empuje")
        col.prop(p, "fuerza")
        sub = col.column(align=True)
        sub.enabled = p.fuerza > 0.0
        sub.prop(p, "vuelta")
        sub.prop(p, "rebote", slider=True)
        if p.fuerza > 0.0 and p.empuje > 0.0:
            col.label(text="los dos a la vez: se suman", icon='INFO')
        elif p.fuerza == 0.0 and p.empuje == 0.0:
            col.label(text="con ambos a cero no reacciona al audio", icon='ERROR')

        col = caja.column(align=True)
        col.label(text="Turbulencia:")
        col.prop(p, "turbulencia")
        sub = col.column(align=True)
        sub.enabled = p.turbulencia > 0.0
        sub.prop(p, "turb_escala")
        sub.prop(p, "turb_velocidad")
        sub.prop(p, "turb_audio", slider=True)

        col = caja.column(align=True)
        col.label(text="Ritmo:")
        if aviso_compas(col, fuente):
            sub = col.column(align=True)
            sub.prop(p, "pulso_onda")
            fila = sub.row()
            fila.enabled = p.pulso_onda > 0.0
            fila.prop(p, "onda_grosor")
            sub.prop(p, "compas_onda")
            fila = sub.row()
            fila.enabled = p.compas_onda > 0.0
            fila.prop(p, "compas_onda_grosor")

        col = caja.column(align=True)
        col.label(text="Aspecto:")
        col.prop(p, "tam_punto")
        col.prop(p, "reaccion_tam")
        col.prop(p, "color_grave")
        col.prop(p, "color_medio")
        col.prop(p, "color_agudo")
        col.prop(p, "brillo")
        col.prop(p, "fondo", slider=True)
        fila = col.row()
        fila.enabled = p.pulso_onda > 0.0 or p.compas_onda > 0.0
        fila.prop(p, "destello")

        caja.separator()
        caja.operator("audioviz.hornear", icon='FILE_TICK')


class AV_PT_atributos(Panel):
    """Chuleta de los atributos que las mallas dejan disponibles al shader."""
    bl_label = "Atributos para el shader"
    bl_idname = "AV_PT_atributos"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Audio Viz"
    bl_parent_id = "AV_PT_plexus"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, contexto):
        d = self.layout
        d.label(text="Nodo Attribute · Tipo: Geometry", icon='NODE')
        d.label(text="Los llevan el plexus Y su objeto de caras.")

        caja = d.box()
        col = caja.column(align=True)
        col.label(text="av_nivel", icon='RNA')
        col.label(text="    0 = graves · 1 = agudos")
        col.label(text="    Donde cae el punto en el espectro.")
        col.label(text="    Fijo: no cambia con la musica.")

        caja = d.box()
        col = caja.column(align=True)
        col.label(text="av_intensidad", icon='RNA')
        col.label(text="    0 = callado · 1 = a tope")
        col.label(text="    Cuanto suena su banda en este fotograma.")
        col.label(text="    Este es el que reacciona.")

        col = d.column(align=True)
        col.label(text="Ambos son de dominio PUNTO, asi que el")
        col.label(text="shader los interpola por la cara: cada")
        col.label(text="triangulo sale degradado entre sus tres")
        col.label(text="esquinas, y cada linea entre sus dos puntas.")

        d.separator()
        col = d.column(align=True)
        col.label(text="Salida a usar del nodo Attribute:", icon='DOT')
        col.label(text="    'Factor' (es un numero, no un color)")

        d.separator()
        caja = d.box()
        col = caja.column(align=True)
        col.label(text="En el ENJAMBRE, tipo: Instancer", icon='ERROR')
        col.label(text="Lo que ves ahi son bolitas instanciadas,")
        col.label(text="y los valores viven en el punto que las")
        col.label(text="instancia, no en la bolita. Con 'Geometry'")
        col.label(text="saldrian todas del mismo color.")
        col.label(text="Ademas lleva un tercero:")
        col.label(text="    av_golpe · 0..1, cuanto le da la onda ahora")


# ---------------------------------------------------------------------------
# REGISTRO
# ---------------------------------------------------------------------------

clases = (AV_AudioAjustes, AV_Ajustes, AV_PaisajeAjustes, AV_EnjambreAjustes,
          AV_PlexusAjustes,
          AV_OT_analizar_audio, AV_OT_importar, AV_OT_tira_sonido,
          AV_OT_ver_analisis, AV_OT_quitar_analisis,
          AV_OT_reaplicar, AV_OT_quitar_suavizado,
          AV_OT_detectar_compas, AV_OT_quitar_compas, AV_OT_medio_tempo,
          AV_OT_crear_barras, AV_OT_crear_led, AV_OT_crear_pulso,
          AV_OT_crear_plexus, AV_OT_crear_paisaje, AV_OT_crear_enjambre,
          AV_OT_hornear, AV_OT_regenerar_puntos, AV_OT_ajustar_distancia,
          AV_OT_seleccionar_plexus, AV_OT_material_unico, AV_OT_limpiar,
          AV_PT_panel, AV_PT_barras, AV_PT_led, AV_PT_pulso, AV_PT_plexus,
          AV_PT_atributos, AV_PT_paisaje, AV_PT_enjambre)


@persistent
def _al_abrir_archivo(_dummy):
    """El espectrograma no se guarda: solo la imagen, que si viaja en el .blend.

    El lienzo del que sale vive en memoria, asi que al abrir otro archivo hay
    que olvidarlo. La imagen sigue viendose; para que la marca del fotograma
    vuelva a moverse basta con pulsar 'Ver el analisis' otra vez.
    """
    _espectros.clear()


_HANDLERS = (
    ("frame_change_pre", "_al_cambiar_fotograma"),
    ("load_post", "_al_abrir_archivo"),
    ("render_pre", "_empieza_render"),
    ("render_post", "_acaba_render"),
    ("render_cancel", "_acaba_render"),
)


def _quitar_handler():
    """Borra los handlers por nombre, no por identidad.

    Al volver a pulsar 'Run Script' se crea una funcion nueva, distinta de la que
    quedo registrada la vez anterior; si buscaramos por identidad no la
    encontrariamos y se irian acumulando handlers duplicados.
    """
    for lista, nombre in _HANDLERS:
        coleccion = getattr(bpy.app.handlers, lista)
        for h in list(coleccion):
            if getattr(h, "__name__", "") == nombre:
                coleccion.remove(h)


def register():
    for c in clases:
        bpy.utils.register_class(c)
    bpy.types.Scene.audioviz = PointerProperty(type=AV_Ajustes)
    # Igual que el plexus, cada fuente de audio lleva sus ajustes pegados al
    # OBJETO y no a la escena: es lo que permite tener varias a la vez.
    bpy.types.Object.audioviz_audio = PointerProperty(type=AV_AudioAjustes)
    bpy.types.Object.audioviz_plex = PointerProperty(type=AV_PlexusAjustes)
    bpy.types.Object.audioviz_paisaje = PointerProperty(type=AV_PaisajeAjustes)
    bpy.types.Object.audioviz_enj = PointerProperty(type=AV_EnjambreAjustes)
    _quitar_handler()
    bpy.app.handlers.frame_change_pre.append(_al_cambiar_fotograma)
    bpy.app.handlers.load_post.append(_al_abrir_archivo)
    bpy.app.handlers.render_pre.append(_empieza_render)
    bpy.app.handlers.render_post.append(_acaba_render)
    bpy.app.handlers.render_cancel.append(_acaba_render)


def unregister():
    _quitar_handler()
    del bpy.types.Object.audioviz_enj
    del bpy.types.Object.audioviz_paisaje
    del bpy.types.Object.audioviz_plex
    del bpy.types.Object.audioviz_audio
    del bpy.types.Scene.audioviz
    for c in reversed(clases):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    # Permite volver a pulsar "Run Script" sin reiniciar Blender.
    try:
        unregister()
    except Exception:
        pass
    register()
    print("Audio Viz registrado. Pulsa N en el visor 3D > pestana 'Audio Viz'.")

