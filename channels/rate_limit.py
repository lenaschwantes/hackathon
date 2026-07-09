"""
Rate limiting por usuário, baseado em contador no Redis.

Protege as chamadas ao motor (Groq/Voyage por trás de `channels.engine`)
contra flood de mensagens: cada usuário tem um número máximo de mensagens
dentro de uma janela de tempo. Quando estoura, a mensagem nem chega a
acionar o motor — o canal responde direto com `MENSAGEM_LIMITE_EXCEDIDO`.
"""

from channels.session import _get_redis

LIMITE_MENSAGENS = 10
JANELA_SEGUNDOS = 60

MENSAGEM_LIMITE_EXCEDIDO = (
    "Você mandou muitas mensagens em pouco tempo. "
    "Espera um minutinho e tenta de novo, por favor."
)


def _chave(user_id: str) -> str:
    return f"ratelimit:{user_id}"


async def permitido(user_id: str) -> bool:
    """
    Verifica se o usuário ainda pode mandar mensagem na janela atual.

    Incrementa o contador a cada chamada; na primeira mensagem da
    janela, define o TTL. Devolve False quando o limite foi estourado
    (a contagem já passou de LIMITE_MENSAGENS).
    """
    r = _get_redis()
    chave = _chave(user_id)
    contagem = await r.incr(chave)
    if contagem == 1:
        await r.expire(chave, JANELA_SEGUNDOS)
    return contagem <= LIMITE_MENSAGENS
