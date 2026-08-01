"""Audio Viz — punto de entrada de la extension.

Todo el codigo vive en blender_audio_viz.py; esto solo lo engancha a Blender.
Se mantienen separados a proposito: ese archivo se puede seguir abriendo en el
editor de texto de Blender y ejecutando con Alt+P para probar un cambio sin
reinstalar nada. (Si lo haces con la extension activada, desactivala antes o
tendras el panel registrado dos veces.)
"""

from . import blender_audio_viz


def register():
    blender_audio_viz.register()


def unregister():
    blender_audio_viz.unregister()
