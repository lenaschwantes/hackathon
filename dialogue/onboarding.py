"""
Contrato explicito pro botao inline da bifurcacao inicial (ver
`channels/engine.py`/`channels/telegram.py`): callback_data que o
Telegram devolve, e o "texto sintetico" que cada um mapeia pro mesmo
fluxo que texto livre equivalente ja segue -- mesmo padrao de
`dialogue/reset.py`'s CALLBACK_REINICIO_*/TEXTO_SINTETICO_*.
"""

CALLBACK_INICIO_BUSCAR = "inicio:buscar"
CALLBACK_INICIO_DUVIDA = "inicio:duvida"
TEXTO_SINTETICO_BUSCAR_CURSO = "quero buscar um curso"
TEXTO_SINTETICO_TENHO_DUVIDA = "tenho uma duvida"

CALLBACK_DUVIDA_GUIA_CURSOS = "duvida:guia_cursos"
CALLBACK_DUVIDA_PRAZOS = "duvida:prazos"
CALLBACK_EDITAL_VER_OUTRO = "edital:ver_outro"
CALLBACK_EDITAL_ENCERRAR = "edital:encerrar"

TEXTO_SINTETICO_GUIA_CURSOS = "quero o guia de cursos"
TEXTO_SINTETICO_DUVIDA_PRAZOS = "duvidas sobre prazos e formas de ingresso"
TEXTO_SINTETICO_VER_OUTRO_EDITAL = "ver outro edital"
TEXTO_SINTETICO_ENCERRAR_DUVIDA = "encerrar duvida"