"""Инструменты my_agent: визуализация данных."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain.tools import tool

from src.tools.tools import ui_image

# Папка для сохранения графиков: <project_root>/outputs/charts/
_CHARTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "charts"


@tool
def plot_chart(
    title: str,
    labels: str,
    values: str,
    chart_type: str = "bar",
    caption: str = "",
) -> str:
    """Строит график по переданным данным и показывает его пользователю в интерфейсе.

    График автоматически появляется в ленте событий Streamlit.
    Поддерживаемые типы: bar (столбчатый), line (линейный), pie (круговой).

    Args:
        title: Заголовок графика.
        labels: Метки категорий через запятую, например: "Янв,Фев,Мар,Апр".
        values: Числовые значения через запятую, например: "10,25,15,30".
        chart_type: Тип графика — "bar", "line" или "pie". По умолчанию "bar".
        caption: Подпись под графиком в интерфейсе (необязательно).

    Returns:
        Сообщение об успехе или описание ошибки.
    """

    label_list = [x.strip() for x in labels.split(",") if x.strip()]
    try:
        value_list = [float(x.strip()) for x in values.split(",") if x.strip()]
    except ValueError as e:
        return f"Ошибка разбора значений: {e}"

    if not label_list or not value_list:
        return "Ошибка: labels и values не должны быть пустыми"

    if len(label_list) != len(value_list):
        return (
            f"Ошибка: количество меток ({len(label_list)}) "
            f"не совпадает с количеством значений ({len(value_list)})"
        )

    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _CHARTS_DIR / f"chart_{int(time.time() * 1000)}.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "bar":
        ax.bar(label_list, value_list, color="#4c8cbf")
        ax.set_xlabel("Категории")
        ax.set_ylabel("Значения")
    elif chart_type == "line":
        ax.plot(label_list, value_list, marker="o", color="#4c8cbf", linewidth=2)
        ax.set_xlabel("Категории")
        ax.set_ylabel("Значения")
    elif chart_type == "pie":
        ax.pie(value_list, labels=label_list, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        plt.close(fig)
        return f"Неизвестный тип графика: '{chart_type}'. Используйте bar, line или pie"

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ui_image(str(filepath), caption or title)
    return f"График '{title}' построен и отображён пользователю"


TOOLS = [plot_chart]
