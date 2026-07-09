import asyncio

from dotenv import load_dotenv
load_dotenv()

from channels.telegram import TelegramAdapter


async def main():
    bot = TelegramAdapter()
    await bot.iniciar()


if __name__ == "__main__":
    asyncio.run(main())