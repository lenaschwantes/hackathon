# Deploy do bot do Telegram

`decifra-bot.service` é a unidade systemd (usuário) usada para manter o
`run_bot.py` rodando permanentemente, com reinício automático em caso de
falha. `WorkingDirectory` e o caminho do `uv` no `ExecStart` assumem a
máquina onde este arquivo foi criado -- ajuste os dois caminhos pro seu
ambiente antes de instalar.

## Instalar

```bash
mkdir -p ~/.config/systemd/user
cp deploy/decifra-bot.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now decifra-bot.service

# opcional, mas necessário pra o serviço sobreviver a logout/reboot sem
# sessão de login ativa:
loginctl enable-linger "$USER"
```

## Operar

```bash
systemctl --user status decifra-bot.service     # ver se está rodando
systemctl --user restart decifra-bot.service    # reiniciar (ex. após deploy novo)
systemctl --user stop decifra-bot.service        # parar
journalctl --user -u decifra-bot.service -f      # logs em tempo real
```
