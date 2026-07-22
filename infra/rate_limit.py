"""
Rate limiting e deduplicação por usuário, baseados em Redis.

Protege as chamadas pagas ao motor (Anthropic/Voyage por trás de
`channels.engine`) contra flood de mensagens: cada usuário tem um
número máximo de mensagens dentro de uma janela de tempo, e mensagens
idênticas repetidas em poucos segundos não são reprocessadas.

Limites vêm de `config.settings` (não hardcoded).

DECISÃO CONSCIENTE DE FALHA ABERTA: se o Redis estiver indisponível,
tanto `permitido()` quanto `eh_duplicada()` LIBERAM a mensagem (retornam
como se estivesse tudo certo) em vez de bloquear. Pro hackathon, é
preferível um flood ocasional a derrubar o bot inteiro porque o Redis
piscou. Em produção, essa escolha deveria ser reavaliada.
"""

import logging

from redis.exceptions import RedisError

from channels.session import _get_redis
from config.settings import settings

logger = logging.getLogger(__name__)

MENSAGEM_LIMITE_EXCEDIDO = (
    "Você mandou muitas mensagens em pouco tempo. "
    "Espera um minutinho e tenta de novo, por favor."
)


def _chave_limite(user_id: str) -> str:
    return f"ratelimit:{user_id}"


def _chave_dedup(user_id: str, texto: str) -> str:
    # hash simples do texto pra não guardar a mensagem inteira como chave
    assinatura = str(hash(texto))
    return f"dedup:{user_id}:{assinatura}"


async def permitido(user_id: str) -> bool:
    """
    Verifica se o usuário ainda pode mandar mensagem na janela atual.

    Incrementa o contador a cada chamada; na primeira mensagem da
    janela, define o TTL. Devolve False quando o limite foi estourado.

    Se o Redis cair, LIBERA (retorna True) -- decisão consciente de
    falha aberta, documentada no topo do módulo.
    """
    try:
        r = _get_redis()
        chave = _chave_limite(user_id)
        contagem = await r.incr(chave)
        if contagem == 1:
            await r.expire(chave, settings.rate_limit_janela_segundos)
        return contagem <= settings.rate_limit_mensagens
    except RedisError as exc:
        logger.error("Rate limit indisponível (Redis), liberando: %s", type(exc).__name__)
        return True


async def eh_duplicada(user_id: str, texto: str) -> bool:
    """
    Detecta se esta mensagem idêntica já chegou do mesmo usuário nos
    últimos `rate_limit_dedup_segundos` segundos.

    Usa SET NX (só grava se a chave ainda não existir) com TTL curto:
    - primeira vez que a mensagem chega -> grava e devolve False
    - repetição dentro da janela -> chave já existe, devolve True

    Se o Redis cair, trata como NÃO duplicada (retorna False) -- mesma
    lógica de falha aberta do resto do módulo.
    """
    try:
        r = _get_redis()
        chave = _chave_dedup(user_id, texto)
        gravou = await r.set(chave, "1", nx=True, ex=settings.rate_limit_dedup_segundos)
        return not gravou
    except RedisError as exc:
        logger.error("Dedup indisponível (Redis), tratando como nova: %s", type(exc).__name__)
        return False