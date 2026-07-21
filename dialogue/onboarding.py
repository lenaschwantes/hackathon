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
