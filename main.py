import os
import sys
import subprocess
import time

# Словарь: "Цифра": ("Имя_файла", "Описание")
SCRIPTS = {
    "1": ("prepare_dataset.py", "✂️  Подготовка датасета (Нарезка + MelSpec)"),
    "2": ("train.py", "🧠  Обучение нейросети"),
    "3": ("debug_matrix.py", "📊  Матрица ошибок (Анализ точности)"),
    "4": ("predict.py", "🎧  Определить жанр трека (Predict)"),
    "5": ("app.py", "🌐  Запуск сервера (FastAPI + Swagger)"),
}


def clear_console():
    """Очищает консоль (работает и на Windows, и на Linux)"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("\n" + "═" * 60)
    print(f"🎹  SOUND UNIVERSE ML: CONTROL CENTER  🎹".center(58))
    print("═" * 60)


def show_menu():
    print_header()
    print("\nВыберите действие:\n")
    for key, (script, desc) in SCRIPTS.items():
        print(f"  [{key}] {desc}")
    print("\n  [0] 🚪 Выход")
    print("\n" + "─" * 60)


def run_script(script_name):
    """Запускает скрипт в текущем окружении Python"""
    if not os.path.exists(script_name):
        print(f"\n❌ ОШИБКА: Файл '{script_name}' не найден!")
        input("\nНажмите Enter, чтобы вернуться в меню...")
        return

    print(f"\n🚀 Запуск {script_name}...\n")
    print("=" * 60 + "\n")

    try:
        # sys.executable гарантирует использование того же python.exe (venv)
        subprocess.run([sys.executable, script_name], check=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Процесс остановлен пользователем.")
    except Exception as e:
        print(f"\n💥 Произошла ошибка: {e}")

    print("\n" + "=" * 60)
    input("\n✅ Готово. Нажмите Enter, чтобы вернуться в меню...")


def main():
    while True:
        # Очищаем экран перед показом меню (чтобы было красиво)
        clear_console()
        show_menu()

        choice = input("Ваш выбор > ").strip().lower()

        if choice in ['0', 'q', 'exit']:
            print("\n👋 bye bye bye...")
            break

        if choice in SCRIPTS:
            script_file, _ = SCRIPTS[choice]
            run_script(script_file)
        else:
            print("\n⚠️ Неверный выбор. Попробуйте цифры от 1 до 5.")
            time.sleep(1.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Выход...")