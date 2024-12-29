import random
from typing import List, Tuple
import pygame
import sys

from config import GAME_CONFIG

class Cell:
    def __init__(self):
        self.is_mine = False
        self.is_revealed = False
        self.is_flagged = False
        self.adjacent_mines = 0

class Minesweeper:
    def __init__(self, width: int, height: int, num_mines: int):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.game_over = False
        self._won = False
        self.board = self._create_board()
        self._place_mines()
        self._calculate_adjacent_mines()
    
    def _create_board(self) -> List[List[Cell]]:
        return [[Cell() for _ in range(self.width)] for _ in range(self.height)]
    
    def _place_mines(self) -> None:
        mines_placed = 0
        while mines_placed < self.num_mines:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if not self.board[y][x].is_mine:
                self.board[y][x].is_mine = True
                mines_placed += 1
    
    def _calculate_adjacent_mines(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                if not self.board[y][x].is_mine:
                    self.board[y][x].adjacent_mines = self._count_adjacent_mines(x, y)
    
    def _count_adjacent_mines(self, x: int, y: int) -> int:
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height and 
                    self.board[new_y][new_x].is_mine):
                    count += 1
        return count

    def reveal(self, x: int, y: int) -> None:
        """Открывает клетку"""
        cell = self.board[y][x]
        
        if cell.is_revealed or cell.is_flagged:
            return
        
        cell.is_revealed = True
        
        if cell.is_mine:
            self.game_over = True
            self.won = False
            return
        
        # Если открыли пустую клетку, открываем соседние
        if cell.adjacent_mines == 0:
            self._reveal_adjacent(x, y)
        
        # После каждого хода проверяем победу
        if self.check_win():
            self.game_over = True
            self.won = True
    
    def _reveal_adjacent(self, x: int, y: int) -> None:
        """Открывает соседние клетки для пустой клетки"""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = x + dx
                new_y = y + dy
                
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height):
                    self.reveal(new_x, new_y)
    
    def toggle_flag(self, x: int, y: int) -> None:
        if not self.game_over and not self.board[y][x].is_revealed:
            self.board[y][x].is_flagged = not self.board[y][x].is_flagged
            self._check_win()
    
    def _check_win(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                cell = self.board[y][x]
                if cell.is_mine:
                    if not cell.is_flagged:
                        return
                else:
                    if not cell.is_revealed:
                        return
        self.won = True
        self.game_over = True

    def check_win(self) -> bool:
        """Проверяет, выиграна ли игра"""
        # Игра выиграна, если все безопасные клетки открыты
        for row in self.board:
            for cell in row:
                # Если клетка без мины и не открыта - игра не выиграна
                if not cell.is_mine and not cell.is_revealed:
                    return False
        return True

    def print_board(self):
        """Отображает текущее состояние игрового поля в консоли"""
        # Печатаем заголовок с номерами столбцов
        print("\n    ", end="")
        for x in range(self.width):
            print(f"{x:2}", end=" ")
        print("\n   ", "-" * (self.width * 3 + 1))
        
        # Печатаем строки поля
        for y in range(self.height):
            print(f"{y:2} |", end=" ")
            for x in range(self.width):
                cell = self.board[y][x]
                if cell.is_revealed:
                    if cell.is_mine:
                        print("* ", end=" ")  # Мина
                    elif cell.adjacent_mines == 0:
                        print(". ", end=" ")  # Пустая клетка
                    else:
                        print(f"{cell.adjacent_mines} ", end=" ")  # Число мин рядом
                elif cell.is_flagged:
                    print("F ", end=" ")  # Флаг
                else:
                    print("□ ", end=" ")  # Закрытая клетка
            print("|")
        
        # Печатаем нижнюю границу
        print("   ", "-" * (self.width * 3 + 1))

    @property
    def won(self) -> bool:
        """Проверяет, выиграна ли игра"""
        if self.game_over:
            # Проверяем, что не открыта ни одна мина
            if not any(cell.is_revealed and cell.is_mine for row in self.board for cell in row):
                # Проверяем, что открыты все безопасные клетки или остались только мины
                safe_cells = sum(1 for row in self.board for cell in row 
                               if not cell.is_mine and not cell.is_revealed)
                return safe_cells == 0
        return False
    
    @won.setter
    def won(self, value: bool):
        """
        Устанавливает значение won
        """
        self._won = value

class MinesweeperGUI:
    def __init__(self, width: int, height: int, num_mines: int):
        pygame.init()
        
        # Константы для отображения
        self.CELL_SIZE = 30
        self.BORDER = 2
        self.COLORS = {
            'REVEALED': (189, 189, 189),
            'UNREVEALED': (220, 220, 220),
            'BORDER': (123, 123, 123),
            'TEXT': (0, 0, 0),
            'MINE': (255, 0, 0),
            'FLAG': (255, 0, 0)
        }
        
        # Инициализация игры
        self.game = Minesweeper(width, height, num_mines)
        
        # Настройка окна
        window_width = width * self.CELL_SIZE
        window_height = height * self.CELL_SIZE
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Сапёр')
        
        # Шрифт для цифр
        self.font = pygame.font.Font(None, 24)
        
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    grid_x = x // self.CELL_SIZE
                    grid_y = y // self.CELL_SIZE
                    
                    # Левый клик - открыть ячейку
                    if event.button == 1:
                        self.game.reveal(grid_x, grid_y)
                    # Правый клик - поставить флажок
                    elif event.button == 3:
                        self.game.toggle_flag(grid_x, grid_y)
            
            self._draw()
            pygame.display.flip()
    
    def _draw(self):
        self.screen.fill(self.COLORS['BORDER'])
        
        for y in range(self.game.height):
            for x in range(self.game.width):
                cell = self.game.board[y][x]
                rect = pygame.Rect(
                    x * self.CELL_SIZE + self.BORDER,
                    y * self.CELL_SIZE + self.BORDER,
                    self.CELL_SIZE - 2 * self.BORDER,
                    self.CELL_SIZE - 2 * self.BORDER
                )
                
                # Отрисовка ячейки
                if cell.is_revealed:
                    pygame.draw.rect(self.screen, self.COLORS['REVEALED'], rect)
                    if cell.is_mine:
                        # Рисуем мину
                        pygame.draw.circle(self.screen, self.COLORS['MINE'],
                                        rect.center, rect.width // 3)
                    elif cell.adjacent_mines > 0:
                        # Рисуем число
                        text = self.font.render(str(cell.adjacent_mines), True, self.COLORS['TEXT'])
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLORS['UNREVEALED'], rect)
                    if cell.is_flagged:
                        # Рисуем флажок
                        points = [
                            (rect.centerx - 5, rect.centery + 5),
                            (rect.centerx - 5, rect.centery - 5),
                            (rect.centerx + 5, rect.centery)
                        ]
                        pygame.draw.polygon(self.screen, self.COLORS['FLAG'], points)

if __name__ == '__main__':
    # Создаем игру 9x9 с 10 минами
    game_gui = MinesweeperGUI(GAME_CONFIG['WIDTH'], GAME_CONFIG['HEIGHT'], GAME_CONFIG['NUM_MINES'])
    game_gui.run()
