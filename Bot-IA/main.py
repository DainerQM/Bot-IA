import discord
from discord.ext import commands
from image_detection import detectar_objetos
import io

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
intents.attachments = True

bot = commands.Bot(command_prefix="/", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user}")

@bot.command()
async def detect(ctx, *, objeto: str):
    if not ctx.message.attachments:
        await ctx.send("Por favor, adjunta una imagen para analizar.")
        return

    attachment = ctx.message.attachments[0]

    if not attachment.content_type.startswith("image"):
        await ctx.send("Por favor, adjunta un archivo de imagen válido.")
        return

    img_data = await attachment.read()

    img, procesada = detectar_objetos(img_data, objeto)

    if procesada:
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        await ctx.send(file=discord.File(output_buffer, "resultado.png"))
    else:
        await ctx.send(f"No se detectó el objeto '{objeto}' en la imagen.")

TOKEN = "TOKEN_DE_DISCORD"
bot.run(TOKEN)
