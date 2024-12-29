import pygame
import time
import os
from tkinter import Tk, filedialog
from main import Minesweeper
from minesweeper_ai import MinesweeperAI
from config import GAME_CONFIG

# Константы для визуализации
CELL_SIZE = 40
MARGIN = 2
COLORS = {
    'BACKGROUND': (200, 200, 200),
    'GRID': (128, 128, 128),
    'COVERED': (169, 169, 169),
    'REVEALED': (220, 220, 220),
    'MINE': (255, 0, 0),
    'FLAG': (255, 165, 0),
    'TEXT': (0, 0, 0),
    'NUMBERS': [
        (0, 0, 255),     # 1 - синий
        (0, 128, 0),     # 2 - зеленый
        (255, 0, 0),     # 3 - красный
        (0, 0, 128),     # 4 - темно-синий
        (128, 0, 0),     # 5 - бордовый
        (0, 128, 128),   # 6 - циан
        (0, 0, 0),       # 7 - черный
        (128, 128, 128)  # 8 - серый
    ]
}

class MinesweeperVisualizer:
    def __init__(self, width: int, height: int, num_mines: int):
        pygame.init()
        
        self.width = width
        self.height = height
        self.window_width = width * CELL_SIZE + (width + 1) * MARGIN
        self.window_height = height * CELL_SIZE + (height + 1) * MARGIN + 50
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Minesweeper AI Visualization")
        
        self.font = pygame.font.Font(None, 36)
        self.game = Minesweeper(width, height, num_mines)
        self.ai = MinesweeperAI(width, height, num_mines)
        
        # Загружаем выбранную пользователем модель
        self.load_model_dialog()
    
    def load_model_dialog(self):
        """Открывает диалог выбора модели"""
        print("\nВыберите модель для загрузки:")
        print("1. Выбрать файл модели")
        print("2. Использовать случайную инициализацию")
        
        choice = input("Ваш выбор (1/2): ").strip()
        
        if choice == "1":
            # Создаем скрытое окно Tkinter для диалога выбора файла
            root = Tk()
            root.withdraw()
            
            # Получаем путь к директории с моделями
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            
            # Открываем диалог выбора файла
            model_path = filedialog.askopenfilename(
                initialdir=models_dir,
                title="Выберите файл модели",
                filetypes=(("PyTorch models", "*.pth"), ("all files", "*.*"))
            )
            
            if model_path:
                try:
                    self.ai.load_model(model_path)
                    print(f"\nУспешно загружена модель: {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"\nОшибка при загрузке модели: {e}")
                    print("Используется случайная инициализация")
            else:
                print("\nВыбор модели отменен. Используется случайная инициализация")
        else:
            print("\nИспользуется случайная инициализация")
    
    def draw_board(self):
        self.screen.fill(COLORS['BACKGROUND'])
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.game.board[y][x]
                rect = pygame.Rect(
                    x * (CELL_SIZE + MARGIN) + MARGIN,
                    y * (CELL_SIZE + MARGIN) + MARGIN,
                    CELL_SIZE,
                    CELL_SIZE
                )
                
                if cell.is_revealed:
                    pygame.draw.rect(self.screen, COLORS['REVEALED'], rect)
                    if cell.is_mine:
                        pygame.draw.rect(self.screen, COLORS['MINE'], rect)
                    elif cell.adjacent_mines > 0:
                        text = self.font.render(str(cell.adjacent_mines), True, 
                                              COLORS['NUMBERS'][cell.adjacent_mines - 1])
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['COVERED'], rect)
                    if cell.is_flagged:
                        pygame.draw.rect(self.screen, COLORS['FLAG'], rect)
        
        # Отображаем статистику
        stats_text = f"Ходов: {self.moves}  Epsilon: {self.epsilon:.3f}"
        text = self.font.render(stats_text, True, COLORS['TEXT'])
        self.screen.blit(text, (10, self.window_height - 40))
        
        pygame.display.flip()
    
    def run(self, delay: float = 0.5, epsilon: float = 0.1):
        """
        Запускает визуализацию игры
        
        Args:
            delay: задержка между ходами в секундах
            epsilon: вероятность случайного хода
        """
        self.moves = 0
        self.epsilon = epsilon
        running = True
        game_over = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Пробел для нового раунда
                        self.game = Minesweeper(self.width, self.height, self.game.num_mines)
                        game_over = False
                        self.moves = 0
                    elif event.key == pygame.K_ESCAPE:  # Escape для выхода
                        running = False
            
            if not game_over:
                try:
                    action = self.ai.choose_action(self.game, epsilon)
                    self.game.reveal(action[0], action[1])
                    self.moves += 1
                    
                    if self.game.game_over:
                        game_over = True
                        result = "Победа!" if self.game.won else "Проигрыш!"
                        print(f"{result} Ходов: {self.moves}")
                
                except ValueError:
                    game_over = True
                    print("Нет доступных ходов")
            
            self.draw_board()
            time.sleep(delay)
        
        pygame.quit()

def main():
    print("\nЗапуск визуализатора Minesweeper AI")
    print("=====================================")
    print(f"Размер поля: {GAME_CONFIG['WIDTH']}x{GAME_CONFIG['HEIGHT']}")
    print(f"Количество мин: {GAME_CONFIG['NUM_MINES']}")
    print("\nУправление:")
    print("- ПРОБЕЛ: новая игра")
    print("- ESC: выход")
    print("=====================================\n")
    
    visualizer = MinesweeperVisualizer(
        GAME_CONFIG['WIDTH'],
        GAME_CONFIG['HEIGHT'],
        GAME_CONFIG['NUM_MINES']
    )
    visualizer.run(delay=0.5, epsilon=0.1)

if __name__ == '__main__':
    main() 