import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from collections import Counter
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    LOC,
    ORG,
    Doc
)

class SuperSmartNLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep AI NLP Analyzer (Natasha Engine)")
        self.root.geometry("900x700")
        
        # Инициализация всех модулей Natasha для "максимального ума"
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_tagger = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        # Премиальная темно-синяя палитра
        self.bg_main = "#0F172A"       # Очень темный синий (Slate)
        self.bg_card = "#1E293B"       # Заполняющий синий
        self.fg_light = "#F8FAFC"      # Почти белый
        self.fg_muted = "#94A3B8"      # Серый текст для подписей
        self.accent_blue = "#38BDF8"   # Неоново-синий
        self.accent_green = "#34D399"  # Изумрудный для кнопок действий

        self.root.configure(bg=self.bg_main)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=self.bg_main)
        self.style.configure('Card.TFrame', background=self.bg_card)
        self.style.configure('TLabel', background=self.bg_main, foreground=self.fg_light, font=('Segoe UI', 10))
        self.style.configure('Muted.TLabel', background=self.bg_main, foreground=self.fg_muted, font=('Segoe UI', 9))
        
        # Стили кнопок
        self.style.configure('Accent.TButton', background=self.accent_blue, foreground=self.bg_main, font=('Segoe UI', 10, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#7DD3FC')])
        
        self.style.configure('Success.TButton', background=self.accent_green, foreground=self.bg_main, font=('Segoe UI', 10, 'bold'))
        self.style.map('Success.TButton', background=[('active', '#6EE7B7')])

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Шапка
        lbl_title = ttk.Label(main_frame, text="🧠 АНАЛИЗАТОР ТЕКСТА НА БАЗЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ", font=('Segoe UI', 14, 'bold'))
        lbl_title.pack(anchor=tk.W, pady=(0, 2))
        lbl_subtitle = ttk.Label(main_frame, text="Извлечение сущностей, синтаксический разбор и статистика", style='Muted.TLabel')
        lbl_subtitle.pack(anchor=tk.W, pady=(0, 15))

        # Поле ввода текста
        self.txt_input = scrolledtext.ScrolledText(
            main_frame, height=7, bg=self.bg_card, fg=self.fg_light,
            insertbackground=self.fg_light, font=('Segoe UI', 11), bd=0, 
            highlightthickness=1, highlightbackground="#334155", relief="flat"
        )
        self.txt_input.pack(fill=tk.X, pady=(0, 10))
        
        default_text = (
            "Директор компании Яндекс Аркадий Волож вчера прилетел в Москву. "
            "Искусственный интеллект стремительно развивает технологии. "
            "Разработчики создают инновационные приложения на языке программирования Python."
        )
        self.txt_input.insert(tk.END, default_text)

        # Панель управления (Кнопки)
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        btn_analyze = ttk.Button(btn_frame, text="🚀 Запустить глубокий ИИ-анализ", style='Accent.TButton', command=self.analyze_text)
        btn_analyze.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_export = ttk.Button(btn_frame, text="💾 Сохранить отчет", style='Success.TButton', command=self.export_report)
        btn_export.pack(side=tk.LEFT, padx=5)

        btn_clear = ttk.Button(btn_frame, text="🧹 Очистить", command=self.clear_all)
        btn_clear.pack(side=tk.LEFT, padx=(5, 0))

        # Поле вывода (Результаты)
        lbl_output = ttk.Label(main_frame, text="Результаты обработки данных:", font=('Segoe UI', 11, 'bold'))
        lbl_output.pack(anchor=tk.W, pady=(0, 5))

        self.txt_output = scrolledtext.ScrolledText(
            main_frame, bg=self.bg_card, fg=self.fg_light,
            font=('Consolas', 10), bd=0, highlightthickness=1, 
            highlightbackground="#334155", relief="flat"
        )
        self.txt_output.pack(fill=tk.BOTH, expand=True)

    def analyze_text(self):
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Внимание", "Пожалуйста, введите текст для анализа!")
            return

        # Пайплайн обработки Natasha
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        doc.parse_syntax(self.syntax_tagger)  # <-- Установлен правильный метод разбора

        # Нормализация сущностей и токенов
        for span in doc.spans:
            span.normalize(self.morph_vocab)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        # Формирование вывода
        self.txt_output.delete("1.0", tk.END)
        
        # 1. Блок Сущностей (NER)
        self.txt_output.insert(tk.END, "=== [1] ИЗВЛЕЧЕННЫЕ ИИ-СУЩНОСТИ (NER) ===\n")
        if doc.spans:
            type_mapping = {PER: "Имя (PER)", ORG: "Организация (ORG)", LOC: "Локация (LOC)"}
            for span in doc.spans:
                label = type_mapping.get(span.type, span.type)
                self.txt_output.insert(tk.END, f"  ● [{label}] {span.normal} (в тексте: '{span.text}')\n")
        else:
            self.txt_output.insert(tk.END, "  Сущности не обнаружены.\n")
        
        # 2. Блок Синтаксических связей
        self.txt_output.insert(tk.END, "\n=== [2] ВЫЯВЛЕННЫЕ СВЯЗИ (КТО -> ЧТО ДЕЛАЕТ) ===\n")
        relations_found = False
        for token in doc.tokens:
            if token.rel == 'nsubj' and token.head_id:
                head_token = next((t for t in doc.tokens if t.id == token.head_id), None)
                if head_token:
                    self.txt_output.insert(tk.END, f"  ✦ Субъект: '{token.text}' ---> Действие: '{head_token.text}'\n")
                    relations_found = True
        if not relations_found:
            self.txt_output.insert(tk.END, "  Не удалось построить явные синтаксические пары.\n")

        # 3. Лингвистическая статистика
        self.txt_output.insert(tk.END, "\n=== [3] СТАТИСТИКА И СТРУКТУРА ТЕКСТА ===\n")
        total_words = len([t for t in doc.tokens if t.pos not in ['PUNCT']])
        self.txt_output.insert(tk.END, f"  ● Всего значимых слов: {total_words}\n")
        
        pos_counts = Counter([t.pos for t in doc.tokens if t.pos not in ['PUNCT']])
        self.txt_output.insert(tk.END, "  ● Топ частей речи в тексте:\n")
        for pos, count in pos_counts.most_common(3):
            self.txt_output.insert(tk.END, f"    - {pos}: {count} раз(а)\n")

    def export_report(self):
        report_content = self.txt_output.get("1.0", tk.END).strip()
        if not report_content or "=== [" not in report_content:
            messagebox.showwarning("Ошибка", "Нет данных для экспорта. Сначала запустите ИИ-анализ.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Сохранить ИИ-отчет"
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=== ОТЧЕТ ГЛУБОКОГО АНАЛИЗА ТЕКСТА ===\n\n")
                f.write(report_content)
            messagebox.showinfo("Успех", "Отчет успешно сохранен!")

    def clear_all(self):
        self.txt_input.delete("1.0", tk.END)
        self.txt_output.delete("1.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SuperSmartNLPApp(root)
    root.mainloop()
