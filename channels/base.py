"""
Interface mínima que todo adaptador de canal precisa seguir.

Um "canal" é o meio pelo qual a pessoa fala com o Decifra (Telegram
hoje, WhatsApp no futuro). Esta interface garante que, seja qual for
o canal, ele sempre recebe (user_id, texto) e devolve texto — nada
além disso. Nenhuma lógica de negócio deve viver aqui nem em nenhum
arquivo que implemente esta interface.
"""

from abc import ABC, abstractmethod


class ChannelAdapter(ABC):
    """
    Contrato que todo canal de mensagem tem que implementar.

    Quem implementa esta classe (ex: TelegramAdapter) é responsável
    apenas por: receber a mensagem crua do canal, extrair o
    user_id e o texto, chamar o motor de resposta injetado, e
    devolver a resposta pelo mesmo canal. Nada mais.
    """

    @abstractmethod
    async def enviar(self, user_id: str, texto: str) -> None:
        """
        Envia uma mensagem de texto para o usuário identificado
        por user_id, usando o canal específico (Telegram, etc).
        """
        raise NotImplementedError

    @abstractmethod
    async def iniciar(self) -> None:
        """
        Liga o canal (ex: começa a escutar mensagens do Telegram).
        Cada implementação decide como fazer isso por dentro.
        """
        raise NotImplementedError