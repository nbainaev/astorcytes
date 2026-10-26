# Работа с кодом

Назначение проекта и команды: [README.md](README.md). Карта компонентов: [ARCHITECTURE.md](ARCHITECTURE.md). Научные определения: [docs/protocol.md](docs/protocol.md).

- Окружение: `uv sync --frozen`; проверка lock-файла: `uv lock --check`.
- Unit-проверки: `uv run python -m unittest discover -s tests`.
- Запуск исследования: `uv run python -m experiments run --workers 30`.
- Подбор: `uv run python -m experiments tune --workers 30 --output results/tuning-study`.
- `src/astrorl` содержит только модели и общие компоненты; он не импортирует `experiments` и не читает файлы конфигурации.
- `experiments` читает конфиги, создаёт модели, управляет запуском и сохраняет результаты.
- Не добавляйте данные расчётов, графики, checkpoint, веса, кэши и архивы в Git.
- Не меняйте бюджеты, seed, карты, метрики и правила остановки при структурных изменениях.
- SNN-архитектуры No Astro/AstroRL/Trace совпадают. RL не обновляет ALIF-ядро; предиктор обновляет его через локальное e-prop.
- MLP использует feedforward local eligibility/direct feedback (DFA). Trace — EMA familiarity bank.
- `configs/conditions.json` и `.csv` должны оставаться согласованными.
- Длительные вычисления выполняются на вычислительном сервере с явно выбранным бюджетом ресурсов.
