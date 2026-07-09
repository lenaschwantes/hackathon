"""
Motor de resposta falso, usado só durante o desenvolvimento.

Este arquivo NUNCA deve importar nada de `retrieval/` ou `recommend/`.
Ele existe pra o bot poder ser testado de ponta a ponta antes de o
motor de verdade (RAG + recomendação) estar pronto.
"""


def responder(user_id: str, texto: str, sessao: dict) -> str:
    """
    Recebe o id do usuário, o texto que ele mandou, e a sessão atual
    (o "estado" da conversa dele, vindo do Redis).

    Por enquanto, só devolve um eco da mensagem. Mais pra frente,
    essa função vai ser substituída pela que de fato consulta o
    RAG e o motor de recomendação — mas a assinatura (os parâmetros
    que ela recebe e o que ela devolve) tem que continuar igual.
    """
    return f"echo: {texto}"