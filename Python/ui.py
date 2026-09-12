from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except Exception:
    pygame = None  # UI is optional

from typing import List

from model import LinearTokenLanguageModel, MAX_TOKENS


class TextBox:
    def __init__(self, rect: 'pygame.Rect', label: str) -> None:
        self.rect = rect
        self.label = label
        self.text = ""
        self.active = False

    def handle(self, event: 'pygame.event.Event') -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_TAB:
                return True
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            elif event.unicode:
                self.text += event.unicode
        return False

    def draw(self, screen: 'pygame.Surface', font: 'pygame.font.Font') -> None:
        border = (240, 240, 240) if self.active else (150, 150, 150)
        pygame.draw.rect(screen, (35, 35, 40), self.rect, border_radius=8)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=8)
        screen.blit(font.render(self.label, True, (220, 220, 220)), (self.rect.x, self.rect.y - 24))
        shown = self.text[-95:]
        screen.blit(font.render(shown, True, (255, 255, 255)), (self.rect.x + 10, self.rect.y + 12))


def button(screen: 'pygame.Surface', font: 'pygame.font.Font', rect: 'pygame.Rect', text: str, on: bool = False) -> None:
    pygame.draw.rect(screen, (70, 95, 130) if on else (55, 55, 65), rect, border_radius=8)
    pygame.draw.rect(screen, (210, 210, 210), rect, 2, border_radius=8)
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label, label.get_rect(center=rect.center))


def _wrap_lines(text: str, width: int = 88) -> List[str]:
    lines: List[str] = []
    for raw_line in (text or "").splitlines() or [""]:
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue
        while len(line) > width:
            split_at = line.rfind(" ", 0, width)
            if split_at <= 0:
                split_at = width
            lines.append(line[:split_at])
            line = line[split_at:].lstrip()
        lines.append(line)
    return lines


def run_gui(model: LinearTokenLanguageModel) -> None:
    if pygame is None:
        raise RuntimeError("pygame is required to run the GUI")

    pygame.init()
    pygame.key.set_repeat(400, 40)
    screen = pygame.display.set_mode((1100, 760))
    pygame.display.set_caption("StellaV Test Console")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    small = pygame.font.SysFont("Arial", 16)
    code_font = pygame.font.SysFont("Courier New", 18)

    from pygame import Rect

    qbox = TextBox(Rect(40, 90, 920, 48), "Question")
    qbox.active = True
    tab_order: List[TextBox] = [qbox]

    last_output = ""
    last_conf = 0.0
    status_text = "Ask about clinical trials, epidemiology, oncology, cardiology, immunology, infection, pharmacology, genomics, neurology, or public health."

    ask_btn = pygame.Rect(40, 165, 140, 44)
    load_btn = pygame.Rect(200, 165, 140, 44)
    save_btn = pygame.Rect(360, 165, 140, 44)
    clear_btn = pygame.Rect(520, 165, 140, 44)

    def ask_current_question() -> None:
        nonlocal last_output, last_conf, status_text
        question = qbox.text.strip()
        if not question:
            status_text = "Type a question first."
            return
        text, last_conf, _ = model.predict(question)
        last_output = text or "I do not have a confident answer yet. Try running bootstrap or learn more data."
        status_text = "Answer ready."

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                model.save()
                running = False

            if event.type == pygame.KEYDOWN and qbox.active and event.key == pygame.K_RETURN:
                ask_current_question()
                continue

            for i, box in enumerate(tab_order):
                if box.handle(event):
                    box.active = False
                    next_box = tab_order[(i + 1) % len(tab_order)]
                    next_box.active = True
                    break

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if ask_btn.collidepoint(pos):
                    ask_current_question()
                elif save_btn.collidepoint(pos):
                    if model.weights is None and model.samples:
                        model.train()
                    model.save()
                    status_text = "Model saved."
                elif load_btn.collidepoint(pos):
                    status_text = "Model loaded." if model.load() else "No saved model found."
                elif clear_btn.collidepoint(pos):
                    qbox.text = ""
                    last_output = ""
                    last_conf = 0.0
                    status_text = "Cleared prompt and output."

        screen.fill((18, 18, 22))
        title = font.render("StellaV Test Console", True, (255, 255, 255))
        screen.blit(title, (40, 30))
        qbox.draw(screen, font)

        button(screen, font, ask_btn, "ASK")
        button(screen, font, load_btn, "LOAD")
        button(screen, font, save_btn, "SAVE")
        button(screen, font, clear_btn, "CLEAR")

        stats = f"Samples={len(model.samples)} | Vocab={len(model.id_to_token)} | MaxContextTokens={MAX_TOKENS} | Linear={'trained' if model.weights is not None else 'fallback only'}"
        screen.blit(small.render(stats, True, (190, 190, 190)), (40, 235))
        screen.blit(small.render(status_text[:120], True, (190, 220, 190)), (40, 262))
        screen.blit(small.render("Enter asks immediately. Stella V prefers ScienceOpen tensor retrieval, then medical knowledge, then linear prediction.", True, (190, 190, 190)), (40, 290))

        panel = pygame.Rect(40, 330, 1020, 380)
        pygame.draw.rect(screen, (30, 30, 36), panel, border_radius=8)
        pygame.draw.rect(screen, (90, 90, 100), panel, 2, border_radius=8)
        output_header = f"Reply  |  confidence={last_conf:.3f}" if last_output else "Reply"
        screen.blit(font.render(output_header, True, (245, 245, 245)), (55, 345))
        y = 380
        for line in _wrap_lines(last_output, width=92):
            if y > panel.bottom - 24:
                break
            screen.blit(code_font.render(line[:100], True, (245, 245, 245)), (55, y))
            y += 22

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
